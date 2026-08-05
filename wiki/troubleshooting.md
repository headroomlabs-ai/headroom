# Troubleshooting Guide

Solutions for common Headroom issues.

---

## Proxy Server Issues

### "Proxy won't start"

**Symptom**: `headroom proxy` fails or hangs.

**Solutions**:

```bash
# 1. Check if port is already in use
lsof -i :8787
# If something is using the port, either kill it or use a different port

# 2. Try a different port
headroom proxy --port 8788

# 3. Check for missing dependencies
pip install "headroom-ai[proxy]"

# 4. Run with request logging
headroom proxy --log-file ~/.headroom/logs/proxy.jsonl --log-messages
```

### "Connection refused" when calling proxy

**Symptom**: `curl: (7) Failed to connect to localhost port 8787`

**Solutions**:

```bash
# 1. Verify proxy is running
curl http://localhost:8787/health

# 2. Check if proxy started on a different port
ps aux | grep headroom

# 3. Check firewall settings (macOS)
sudo pfctl -s rules | grep 8787
```

### "Upstream rejects a beta token the client no longer sends"

**Symptom**: The upstream API returns an error referencing a beta feature (`anthropic-beta` header) even though the client is no longer sending that header.

**Cause**: Headroom's `SessionBetaTracker` re-injects any `anthropic-beta` token seen earlier in the same session to preserve prefix-cache stability. Once a token is in the tracker it persists for the rest of the session. Stopping the token on the client side alone is not sufficient.

**Solution**: Set `HEADROOM_BETA_HEADER_STICKY=disabled` to pass the client's header value verbatim without accumulation:

```bash
export HEADROOM_BETA_HEADER_STICKY=disabled
headroom proxy ...
```

Alternatively, restarting the proxy process clears the in-memory tracker. See [Session Beta Header Tracking](configuration.md#session-beta-header-tracking) for details.

---

### "Proxy returns errors for some requests"

**Symptom**: Some requests work, others fail with 502/503.

**Solutions**:

```bash
# 1. Check proxy logs for the actual error
headroom proxy --log-file ~/.headroom/logs/proxy.jsonl --log-messages

# 2. Verify API key is set
echo $OPENAI_API_KEY  # or ANTHROPIC_API_KEY

# 3. Test the underlying API directly
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

## SDK Issues

### "No token savings"

**Symptom**: `stats['session']['tokens_saved_total']` is 0.

**Diagnosis**:

```python
# 1. Check mode
stats = client.get_stats()
print(f"Mode: {stats['config']['mode']}")  # Should be "optimize"

# 2. Check transforms are enabled
print(f"SmartCrusher: {stats['transforms']['smart_crusher_enabled']}")

# 3. Check if content meets threshold
# SmartCrusher only compresses tool outputs > 200 tokens by default
```

**Solutions**:

```python
# 1. Ensure mode is "optimize"
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    default_mode="optimize",  # NOT "audit"
)

# 2. Or override per-request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    headroom_mode="optimize",
)

# 3. Lower the compression threshold
config = HeadroomConfig()
config.smart_crusher.min_tokens_to_crush = 100  # Default is 200
```

**Why It Might Be 0**:
- Mode is "audit" (observation only)
- Messages don't contain tool outputs
- Tool outputs are below the token threshold
- Data isn't compressible (high uniqueness)

### "Compression too aggressive"

**Symptom**: LLM responses are missing information that was in tool outputs.

**Solutions**:

```python
# 1. Keep more items
config = HeadroomConfig()
config.smart_crusher.max_items_after_crush = 50  # Default: 15

# 2. Skip compression for specific tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    headroom_tool_profiles={
        "important_tool": {"skip_compression": True},
    },
)

# 3. Disable SmartCrusher entirely
config.smart_crusher.enabled = False
```

### "High latency"

**Symptom**: Requests take longer than expected.

**Diagnosis**:

```python
import time
import logging

logging.basicConfig(level=logging.DEBUG)

start = time.time()
response = client.chat.completions.create(...)
print(f"Total time: {time.time() - start:.2f}s")

# Check logs for:
# - "SmartCrusher" timing
# - "EmbeddingScorer" timing (slow if using embeddings)
```

**Solutions**:

```python
# 1. Use BM25 instead of embeddings (faster)
config = HeadroomConfig()
config.smart_crusher.relevance.tier = "bm25"  # Default may use embeddings

# 2. Increase threshold to skip small payloads
config.smart_crusher.min_tokens_to_crush = 500

# 3. Disable transforms you don't need
config.cache_aligner.enabled = False
config.rolling_window.enabled = False
```

### "ValidationError on setup"

**Symptom**: `validate_setup()` returns errors.

**Common Issues**:

```python
result = client.validate_setup()
print(result)

# Provider error:
# {"provider": {"ok": False, "error": "No API key"}}
# → Set OPENAI_API_KEY or pass api_key to OpenAI()

# Storage error:
# {"storage": {"ok": False, "error": "unable to open database"}}
# → Check path permissions, use :memory: for testing

# Config error:
# {"config": {"ok": False, "error": "Invalid mode"}}
# → Use "audit" or "optimize" only
```

**Solutions**:

```python
# 1. For testing, use in-memory storage
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    store_url="sqlite:///:memory:",  # No file created
)

# 2. For temp directory storage
import tempfile
import os
db_path = os.path.join(tempfile.gettempdir(), "headroom.db")
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    store_url=f"sqlite:///{db_path}",
)
```

---

## Import/Installation Issues

### "pip install fails with C++ compilation error"

**Symptom**: Installation fails with an error like:

```
RuntimeError: Unsupported compiler -- at least C++11 support is needed!
ERROR: Failed building wheel for hnswlib
```

**Cause**: `headroom-ai` depends on `hnswlib`, a C++ extension that must be compiled from source. Slim environments (Docker slim images, minimal CI runners) lack the required build tools.

**Solutions**:

```bash
# Linux / Debian-based (including Docker)
apt-get install -y build-essential && pip install headroom-ai

# macOS (Xcode command line tools)
xcode-select --install && pip install headroom-ai
```

In a Dockerfile, install and remove build tools in one layer to keep the image slim:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install "headroom-ai[proxy]" \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
```

---

### "ModuleNotFoundError: No module named 'headroom'"

```bash
# 1. Check it's installed in the right environment
pip show headroom-ai

# 2. If using virtual environment, ensure it's activated
source venv/bin/activate  # or equivalent

# 3. Reinstall
pip install --upgrade headroom-ai
```

### "ImportError: cannot import name 'X' from 'headroom'"

```python
# Check available imports
import headroom
print(dir(headroom))

# Common imports:
from headroom import (
    HeadroomClient,
    OpenAIProvider,
    AnthropicProvider,
    HeadroomConfig,
    # Exceptions
    HeadroomError,
    ConfigurationError,
    ProviderError,
)
```

### "Missing optional dependency"

```bash
# For proxy server
pip install "headroom-ai[proxy]"

# For embedding-based relevance scoring
pip install "headroom-ai[relevance]"

# For everything
pip install "headroom-ai[all]"
```

---

## Provider-Specific Issues

### OpenAI: "Invalid API key"

```python
from openai import OpenAI
import os

# Ensure key is set
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

client = HeadroomClient(
    original_client=OpenAI(api_key=api_key),
    provider=OpenAIProvider(),
)
```

### Anthropic: "Authentication error"

```python
from anthropic import Anthropic
import os

api_key = os.environ.get("ANTHROPIC_API_KEY")
client = HeadroomClient(
    original_client=Anthropic(api_key=api_key),
    provider=AnthropicProvider(),
)
```

### "Unknown model" warnings

```python
# For custom/fine-tuned models, specify context limit
client = HeadroomClient(
    original_client=OpenAI(),
    provider=OpenAIProvider(),
    model_context_limits={
        "ft:gpt-4o-2024-08-06:my-org::abc123": 128000,
        "my-custom-model": 32000,
    },
)
```

---

## Debugging Techniques

### Enable Full Logging

```python
import logging

# See everything
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Or just Headroom logs
logging.getLogger("headroom").setLevel(logging.DEBUG)
```

### Inspect Transform Results

```python
# Use simulate to see what would happen
plan = client.chat.completions.simulate(
    model="gpt-4o",
    messages=messages,
)

print(f"Tokens: {plan.tokens_before} -> {plan.tokens_after}")
print(f"Transforms: {plan.transforms}")
print(f"Waste signals: {plan.waste_signals}")

# See the actual optimized messages
import json
print(json.dumps(plan.messages_optimized, indent=2))
```

### Check Storage Contents

```python
from datetime import datetime, timedelta

# Get recent metrics
metrics = client.get_metrics(
    start_time=datetime.utcnow() - timedelta(hours=1),
    limit=10,
)

for m in metrics:
    print(f"{m.timestamp}: {m.tokens_input_before} -> {m.tokens_input_after}")
    print(f"  Transforms: {m.transforms_applied}")
    if m.error:
        print(f"  ERROR: {m.error}")
```

### Manual Transform Testing

```python
from headroom import SmartCrusher, Tokenizer
from headroom.config import SmartCrusherConfig
import json

# Test compression directly
config = SmartCrusherConfig()
crusher = SmartCrusher(config)
tokenizer = Tokenizer()

messages = [
    {"role": "tool", "content": json.dumps({"items": list(range(100))}), "tool_call_id": "1"}
]

result = crusher.apply(messages, tokenizer)
print(f"Tokens: {result.tokens_before} -> {result.tokens_after}")
print(f"Compressed content: {result.messages[0]['content'][:200]}...")
```

---

### "Native detector crashes with illegal instruction"

On some older or virtualized x86_64 CPUs, AVX2 may be unavailable. The
Magika/ONNX Runtime detector can require AVX2 through its precompiled runtime
binary. Headroom skips that detector tier on x86/x86_64 hosts without AVX2 and
falls back to non-Magika detection tiers instead of crashing.

If native startup still fails on an older CPU, set:

```bash
export HEADROOM_REQUIRE_RUST_CORE=false
```

---

## Error Reference

| Exception | Meaning | Solution |
|-----------|---------|----------|
| `ConfigurationError` | Invalid config values | Check config parameters |
| `ProviderError` | Provider issue (unknown model, etc.) | Set model_context_limits |
| `StorageError` | Database issue | Check path/permissions |
| `CompressionError` | Compression failed | Rare - check data format |
| `TokenizationError` | Token counting failed | Check model name |
| `ValidationError` | Setup validation failed | Run validate_setup() |

### Handling Errors

```python
from headroom import (
    HeadroomClient,
    HeadroomError,
    ConfigurationError,
    StorageError,
)

try:
    client = HeadroomClient(...)
    response = client.chat.completions.create(...)
except ConfigurationError as e:
    print(f"Config issue: {e}")
    print(f"Details: {e.details}")
except StorageError as e:
    print(f"Storage issue: {e}")
    # Headroom continues to work, just without metrics persistence
except HeadroomError as e:
    print(f"Headroom error: {e}")
```

---

## Getting Help

1. **Enable debug logging** and check the output
2. **Use simulate()** to see what transforms would apply
3. **Check validate_setup()** for configuration issues
4. **File an issue** at https://github.com/headroom-sdk/headroom/issues

When filing an issue, include:
- Headroom version (`pip show headroom`)
- Python version
- Provider (OpenAI/Anthropic)
- Debug log output
- Minimal reproduction code
