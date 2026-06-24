# MCP Server — Context Engineering Toolkit

Headroom's MCP server exposes **compression, retrieval, and observability** as tools that any MCP-compatible AI coding tool can use — Claude Code, Cursor, Codex, and more.

## Quick Start

```bash
# Install (MCP is included with proxy, or standalone)
pip install "headroom-ai[proxy]"    # Proxy + MCP tools
pip install "headroom-ai[mcp]"      # MCP tools only (lightweight)

# Register with Claude Code (one-time)
headroom mcp install

# Start Claude Code — it now has headroom tools!
claude
```

That's it. Claude Code can now compress content on demand, retrieve originals, and check session stats — **no proxy required**.

For automatic compression of ALL traffic, also run the proxy:

```bash
# Terminal 1
headroom proxy

# Terminal 2
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

## Tools

The MCP server provides three tools:

### headroom_compress

Compress content on demand. The LLM calls this when it wants to shrink large content before reasoning over it.

```
Tool: headroom_compress

Parameters:
  - content (required): Text to compress (files, JSON, logs, search results, etc.)

Returns:
  - compressed: Compressed text
  - hash: Key for retrieving the original later
  - original_tokens / compressed_tokens / savings_percent
  - transforms: Which compression algorithms were applied
```

Example — Claude reads a large file, then compresses it:

```
Claude: Let me compress this large output to save context space.

→ headroom_compress(content="[5000 lines of grep results...]")

← {
    "compressed": "[key matches with context...]",
    "hash": "a1b2c3d4e5f6...",
    "original_tokens": 12000,
    "compressed_tokens": 3200,
    "savings_percent": 73.3,
    "transforms": ["router:search:0.27"]
  }
```

The original is stored locally for the session (1-hour TTL). If Claude needs the full content later, it calls `headroom_retrieve`.

### headroom_retrieve

Retrieve original uncompressed content by hash.

```
Tool: headroom_retrieve

Parameters:
  - hash (required): Hash key from compression
  - query (optional): Search within the original to return only matching items

Returns:
  - original_content (full retrieval) or results (search)
  - source: "local" or "proxy"
```

Retrieval checks the local store first (content compressed via `headroom_compress`), then falls back to the proxy's store (content compressed automatically by the proxy). Hashes from either source work transparently.

### headroom_stats

Session compression statistics — including sub-agent stats and proxy cache info.

```
Tool: headroom_stats

Returns:
  - compressions, retrievals, tokens_saved, savings_percent
  - estimated_cost_saved_usd
  - recent_events (last 10 compression/retrieval events)
  - sub_agents (stats from sub-agent MCP instances, if any)
  - combined (main + sub-agent totals)
  - proxy (request count, cache hits, cost saved — if proxy is running)
```

Sub-agent stats are aggregated via a shared stats file at
`${HEADROOM_WORKSPACE_DIR}/session_stats.jsonl` (default
`~/.headroom/session_stats.jsonl` — see the
[Filesystem Contract](filesystem-contract.md)). Each MCP server instance
(main session and sub-agents) writes events there, and `headroom_stats`
reads across all of them.

## Architecture

### MCP Only (no proxy)

```
┌─────────────────────────────────────────────┐
│  Claude Code / Cursor / Codex               │
│                                              │
│  LLM calls headroom_compress on demand       │
│  ↓                                           │
│  Compression happens locally in MCP process  │
│  Original stored in local CompressionStore   │
│  ↓                                           │
│  LLM calls headroom_retrieve when needed     │
└─────────────────────────────────────────────┘
```

### MCP + Proxy (full setup)

```
┌─────────────────────────────────────────────┐
│  Claude Code                                 │
│                                              │
│  1. Sends request ──→ Proxy (auto-compress)  │
│  2. Gets response with compressed outputs    │
│  3. Can call headroom_compress for more      │
│  4. headroom_retrieve checks:                │
│     local store → proxy store                │
└──────────────────┬──────────────────────────┘
                   │ MCP (stdio)
                   ▼
┌─────────────────────────────────────────────┐
│  Headroom MCP Server                         │
│  ├── headroom_compress  (local compression)  │
│  ├── headroom_retrieve  (local + proxy)      │
│  └── headroom_stats     (aggregated stats)   │
└─────────────────────────────────────────────┘
```

No double-compression: the proxy compresses at the HTTP level (before the LLM sees content). MCP tools operate after the LLM receives content. They don't touch the same data.

## CLI Commands

### Install

```bash
headroom mcp install                              # Default setup
headroom mcp install --proxy-url http://host:9000  # Custom proxy URL
headroom mcp install --force                       # Overwrite existing
```

### Status

```bash
headroom mcp status
```

```
Headroom MCP Status
========================================
MCP SDK:        ✓ Installed
Claude Config:  ✓ Configured
                /Users/you/.claude/mcp.json
Proxy URL:      http://127.0.0.1:8787
Proxy Status:   ✓ Running at http://127.0.0.1:8787
```

### Uninstall

```bash
headroom mcp uninstall
```

### Debug

```bash
headroom mcp serve --debug
```

## Cross-Tool Compatibility

The MCP server works with any MCP-compatible host:

| Tool | MCP Support | Setup |
|------|-------------|-------|
| Claude Code | Native | `headroom mcp install` |
| Cursor | Supported | Add to Cursor MCP settings |
| Codex | If supported | Configure MCP server |
| Any MCP host | Yes | Point to `headroom mcp serve` |

## Troubleshooting

### "MCP SDK not installed"

```bash
pip install "headroom-ai[mcp]"
```

### "Proxy not running" (when using proxy features)

```bash
headroom proxy  # In another terminal
```

### "Entry not found or expired"

- Content compressed via `headroom_compress`: stored for 1 hour (session TTL)
- Content compressed by the proxy: stored for 5 minutes (proxy TTL)
- The proxy must be running for proxy-compressed content

### Claude doesn't see headroom tools

1. Check: `headroom mcp status`
2. Restart Claude Code after installing MCP
3. Verify with `/mcp` in Claude Code — should show 3 headroom tools

### Sub-agent stats not showing

Sub-agent stats appear in `headroom_stats` only after sub-agents have run compressions. The shared stats file is at `${HEADROOM_WORKSPACE_DIR}/session_stats.jsonl` (defaults to `~/.headroom/session_stats.jsonl`).
