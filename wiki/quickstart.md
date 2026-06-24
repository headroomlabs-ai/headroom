# Quickstart Guide

Get Headroom running in 5 minutes with these copy-paste examples.

---

## Installation

**Python:**

```bash
# Core only (minimal dependencies)
pip install headroom-ai

# With proxy server
pip install "headroom-ai[proxy]"

# Everything
pip install "headroom-ai[all]"
```

**TypeScript / Node.js:**

```bash
npm install headroom-ai
```

**Docker-native:**

```bash
curl -fsSL https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.sh | bash
```

See [Docker-native install](docker-install.md) if you want Docker to provide the Headroom runtime while your agent CLIs stay on the host.

**Persistent background runtime:**

```bash
headroom install apply --preset persistent-service --providers auto
```

See [Persistent Installs](persistent-installs.md) if you want Headroom to stay up in the background and be reused by `wrap`.

---

## Option 1: Proxy Server (Zero Code Changes)

The fastest way to start saving tokens. Works with any OpenAI-compatible client.

### Step 1: Start the Proxy

```bash
headroom proxy --port 8787
```

### Step 2: Verify It's Running

```bash
curl http://localhost:8787/health
# Expected: {"status":"healthy","ready":true,"config":{"backend":"anthropic",...},...}
```

### Step 3: Point Your Client

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# GitHub Copilot CLI (default Anthropic-style proxy route)
headroom wrap copilot -- --model claude-sonnet-4-20250514

# Cursor / Continue / any OpenAI client
OPENAI_BASE_URL=http://localhost:8787/v1 your-app

# Python
export OPENAI_BASE_URL=http://localhost:8787/v1
python your_script.py
```

### Step 4: Check Savings

```bash
curl http://localhost:8787/stats
# {"requests_total": 42, "tokens_saved_total": 125000, ...}
```

---

## Option 2: Python SDK

Wrap your existing client for fine-grained control.

### Basic Example

```python
from headroom import HeadroomClient, OpenAIProvider
from openai import OpenAI

# Create wrapped client
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    default_mode="optimize",
)

# Use exactly like OpenAI client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)

print(response.choices[0].message.content)

# Check what happened
stats = client.get_stats()
print(f"Tokens saved: {stats['session']['tokens_saved_total']}")
```

### With Tool Outputs (Where Savings Happen)

```python
from headroom import HeadroomClient, OpenAIProvider
from openai import OpenAI
import json

client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    default_mode="optimize",
)

# Simulate a conversation with large tool outputs
messages = [
    {"role": "system", "content": "You analyze search results."},
    {"role": "user", "content": "Search for Python tutorials."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "python"}'},
        }],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        # This is where Headroom shines - compressing large outputs
        "content": json.dumps({
            "results": [{"title": f"Result {i}", "score": 100-i} for i in range(500)]
        }),
    },
    {"role": "user", "content": "What are the top 3 results?"},
]

# Headroom compresses the 500 results to ~20, keeping the most relevant
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)

print(response.choices[0].message.content)
```

### Simulate Before Sending

Preview optimizations without making an API call:

```python
# See what would happen without calling the API
plan = client.chat.completions.simulate(
    model="gpt-4o",
    messages=messages,
)

print(f"Tokens before: {plan.tokens_before}")
print(f"Tokens after: {plan.tokens_after}")
print(f"Would save: {plan.tokens_saved} tokens ({plan.tokens_saved/plan.tokens_before*100:.0f}%)")
print(f"Transforms: {plan.transforms}")
print(f"Estimated savings: {plan.estimated_savings}")
```

---

## Option 3: Anthropic SDK

```python
from headroom import HeadroomClient, AnthropicProvider
from anthropic import Anthropic

client = HeadroomClient(
    original_client=Anthropic(),
    provider=AnthropicProvider(),
    default_mode="optimize",
)

# Use Anthropic-style API
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"},
    ],
)

print(response.content[0].text)
```

---

## Verify It's Working

### Method 1: Enable Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now you'll see:
# INFO:headroom.transforms.pipeline:Pipeline complete: 45000 -> 4500 tokens (saved 40500, 90.0% reduction)
# INFO:headroom.transforms.smart_crusher:SmartCrusher: keeping 15 of 500 items
```

### Method 2: Check Session Stats

```python
stats = client.get_stats()
print(stats)
# {
#   "session": {"requests_total": 10, "tokens_saved_total": 5000, ...},
#   "config": {"mode": "optimize", "provider": "openai", ...},
#   "transforms": {"smart_crusher_enabled": True, ...}
# }
```

### Method 3: Validate Setup

```python
result = client.validate_setup()
if not result["valid"]:
    print("Setup issues:", result)
else:
    print("Setup OK!")
    print(f"Provider: {result['provider']['name']}")
    print(f"Storage: {result['storage']['url']}")
```

---

## Common Configuration

### Adjust Compression

```python
from headroom import HeadroomClient, OpenAIProvider, HeadroomConfig

config = HeadroomConfig()

# Keep more items after compression (default: 15)
config.smart_crusher.max_items_after_crush = 30

# Only compress if tool output has > 500 tokens (default: 200)
config.smart_crusher.min_tokens_to_crush = 500

client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    config=config,  # Pass custom config
    default_mode="optimize",
)
```

### Skip Compression for Specific Tools

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    headroom_tool_profiles={
        "database_query": {"skip_compression": True},  # Never compress
        "search": {"max_items": 50},  # Keep more items
    },
)
```

### Audit Mode (Observe Only)

```python
# Start in audit mode - see what WOULD be optimized
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    default_mode="audit",  # No modifications, just logging
)

# Override per-request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    headroom_mode="optimize",  # Enable for this request only
)
```

---

## What Gets Optimized?

| Content Type | What Headroom Does | Typical Savings |
|--------------|-------------------|-----------------|
| **Tool outputs with lists** | Keeps errors, anomalies, high-score items | 70-90% |
| **Repeated search results** | Deduplicates and samples | 60-80% |
| **Long conversations** | Drops old turns, keeps recent | 40-60% |
| **System prompts with dates** | Stabilizes for cache hits | Cache savings |

---

## Next Steps

- **[Configuration Reference](api.md)** - All configuration options
- **[Transform Reference](transforms.md)** - How each transform works
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Examples](../examples/)** - More complete examples

---

## Quick Troubleshooting

### "No token savings"

```python
# 1. Check mode
stats = client.get_stats()
print(stats["config"]["mode"])  # Should be "optimize"

# 2. Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG)
```

### "High latency"

```python
# Use BM25 instead of embeddings for faster relevance scoring
config.smart_crusher.relevance.tier = "bm25"
```

### "Compression too aggressive"

```python
# Keep more items
config.smart_crusher.max_items_after_crush = 50
```

See [Troubleshooting Guide](troubleshooting.md) for more solutions.
