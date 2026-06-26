# Getting Started with Headroom

This guide will help you get up and running with Headroom in under 5 minutes.

## Installation

**Python:**

```bash
# Core package (minimal dependencies)
pip install headroom-ai

# With proxy server
pip install headroom-ai[proxy]

# With semantic relevance (for smarter compression)
pip install headroom-ai[relevance]

# Everything
pip install headroom-ai[all]
```

**TypeScript / Node.js:**

```bash
npm install headroom-ai
```

**Docker-native:**

```bash
curl -fsSL https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.sh | bash
```

PowerShell:

```powershell
irm https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.ps1 | iex
```

See [Docker-native install](docker-install.md) for wrapper behavior, compose usage, and host-integrated `wrap` flows.

If you want Headroom to stay up in the background and automatically serve supported tools, use [Persistent Installs](persistent-installs.md):

```bash
headroom install apply --preset persistent-service --providers auto
```

## Quick Start: Proxy Mode (Recommended)

The easiest way to use Headroom is as a proxy server:

```bash
# Start the proxy
headroom proxy --port 8787
```

Then point your LLM client at it:

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# GitHub Copilot CLI (default Anthropic-style proxy route)
headroom wrap copilot -- --model claude-sonnet-4-20250514

# OpenAI-compatible clients
OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

That's it! All your requests now go through Headroom and get optimized automatically.

## Quick Start: Python SDK

If you want programmatic control:

```python
from headroom import HeadroomClient
from openai import OpenAI

# Create a wrapped client
client = HeadroomClient(
    original_client=OpenAI(),
    default_mode="optimize",
)

# Use exactly like the original
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
```

## Modes

### Audit Mode

Observe without modifying:

```python
client = HeadroomClient(
    original_client=OpenAI(),
    default_mode="audit",
)
# Logs metrics but doesn't change requests
```

### Optimize Mode

Apply transforms to reduce tokens:

```python
client = HeadroomClient(
    original_client=OpenAI(),
    default_mode="optimize",
)
# Compresses tool outputs, aligns cache prefixes, etc.
```

### Simulate Mode

Preview what optimizations would do:

```python
plan = client.chat.completions.simulate(
    model="gpt-4o",
    messages=[...],
)
print(f"Would save {plan.tokens_saved} tokens")
print(f"Transforms: {plan.transforms_applied}")
```

## Next Steps

- [Proxy Server Documentation](proxy.md) - Configure the proxy
- [Transforms Reference](transforms.md) - Understand each transform
- [API Reference](api.md) - Full API documentation
