# MCP + Headroom Demo

Real-world demonstration of Headroom compression on MCP (Model Context Protocol) tool outputs.

## Quick Start

```bash
# Show compression on mock MCP tool outputs (no API key needed)
PYTHONPATH=. python -m examples.mcp_demo.show_compression

# Show the minimal BEFORE/AFTER code change for MCP integration
PYTHONPATH=. python -m examples.mcp_demo.show_before_after

# Run the full agent evaluation (requires OPENAI_API_KEY)
export OPENAI_API_KEY='your-key-here'
PYTHONPATH=. python -m examples.mcp_demo.run_agent_eval
```

## What each script does

| Script | Purpose |
|--------|---------|
| `show_compression.py` | Compresses mock MCP tool results (Slack search, database queries, GitHub issues, log analysis) via `compress_tool_result_with_metrics` and prints before/after sizes. |
| `show_before_after.py` | Prints the minimal code diff needed to add Headroom compression to MCP tool outputs in your host application. |
| `run_agent_eval.py` | Simulates an agent with multiple MCP tools and tests whether compression preserves the information needed to answer correctly. Deterministic test-data generators keep the eval reproducible. |
| `mock_mcp_servers.py` | Test-data generators simulating real MCP server outputs. |

## Key imports

```python
from headroom.integrations.mcp import compress_tool_result_with_metrics
from headroom.providers import OpenAIProvider
```
