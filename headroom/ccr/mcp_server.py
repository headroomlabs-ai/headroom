"""Headroom MCP Server — Context engineering toolkit for AI coding tools.

Exposes Headroom's compression, retrieval, and observability as MCP tools
that any MCP-compatible host (Claude Code, Cursor, Codex, etc.) can use.

Tools:
    headroom_compress   — Compress content on demand (no proxy needed)
    headroom_retrieve   — Retrieve original uncompressed content by hash
    headroom_stats      — Session compression statistics

Usage:
    # As standalone server (stdio transport, called by AI coding tools)
    headroom mcp serve

    # Add to Claude Code
    headroom mcp install

When running standalone (no proxy), compression and retrieval happen locally
in this process. When a proxy is running, retrieval can also fetch from the
proxy's compression store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headroom import paths as _paths

# fcntl is Unix-only; on Windows we skip file locking (stats are best-effort).
# Keep the module typed as Any so Windows mypy runs don't try to resolve Unix-only attrs.
fcntl: Any = None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

# Try to import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None  # type: ignore[assignment,misc]
    stdio_server = None  # type: ignore[assignment]

# Try to import httpx for proxy communication
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

CCR_TOOL_NAME = "headroom_retrieve"
COMPRESS_TOOL_NAME = "headroom_compress"
STATS_TOOL_NAME = "headroom_stats"
READ_TOOL_NAME = "headroom_read"

logger = logging.getLogger("headroom.ccr.mcp")

# Feature flag: enable headroom_read tool (file read caching via CCR)
# Set HEADROOM_MCP_READ=on to enable
_READ_ENABLED = os.environ.get("HEADROOM_MCP_READ", "off").lower().strip() in (
    "on",
    "true",
    "1",
    "yes",
    "enabled",
)

DEFAULT_PROXY_URL = os.environ.get("HEADROOM_PROXY_URL", "http://127.0.0.1:8787")


def _format_session_summary(summary: dict[str, Any], local_stats: dict[str, Any]) -> str:
    """Format the proxy summary + local MCP stats into clean readable text."""
    lines: list[str] = []
    lines.append("Headroom Session Summary")
    lines.append("=" * 40)

    mode = summary.get("mode", "token")
    api_reqs = summary.get("api_requests", 0)
    model = summary.get("primary_model", "unknown")
    lines.append(f"Mode: {mode} | {api_reqs} API requests | {model}")
    lines.append("")

    # Compression section
    comp = summary.get("compression", {})
    n_compressed = comp.get("requests_compressed", 0)
    if n_compressed > 0:
        lines.append(f"Compression ({n_compressed} requests compressed):")
        lines.append(f"  Avg compression:  {comp.get('avg_compression_pct', 0)}%")
        best = comp.get("best_compression_pct", 0)
        detail = comp.get("best_detail", "")
        if best > 0:
            lines.append(f"  Best compression: {best}% ({detail})")
        removed = comp.get("total_tokens_removed", 0)
        lines.append(f"  Tokens removed:   {removed:,}")
    else:
        lines.append("Compression: no requests compressed yet")
    lines.append("")

    # Uncompressed reasons
    uncomp = summary.get("uncompressed_requests", {})
    if uncomp:
        total_uncomp = sum(uncomp.values())
        lines.append(f"Uncompressed requests ({total_uncomp}):")
        reason_labels = {
            "prefix_frozen": "Prefix-frozen (cached by provider)",
            "too_small": "Too small (< 500 tokens)",
            "passthrough": "Passthrough (token counting)",
            "no_compressible_content": "No compressible content (user/assistant only)",
        }
        for key, count in uncomp.items():
            label = reason_labels.get(key, key)
            lines.append(f"  {label}: {count}")
        lines.append("")

    # Cost section
    cost = summary.get("cost", {})
    without = cost.get("without_headroom_usd", 0)
    with_hr = cost.get("with_headroom_usd", 0)
    saved = cost.get("total_saved_usd", 0)
    pct = cost.get("savings_pct", 0)
    if without > 0:
        lines.append("Cost Impact:")
        lines.append(f"  Without Headroom:  ${without:.2f}")
        lines.append(f"  With Headroom:     ${with_hr:.2f}")
        lines.append(f"  You saved:         ${saved:.2f} ({pct}%)")
        breakdown = cost.get("breakdown", {})
        cache_s = breakdown.get("cache_savings_usd", 0)
        comp_s = breakdown.get("compression_savings_usd", 0)
        if cache_s > 0 or comp_s > 0:
            lines.append(f"    Cache savings:       ${cache_s:.2f}")
            lines.append(f"    Compression savings: ${comp_s:.2f}")
        lines.append("")

    # MCP-local stats (compressions done by MCP tool directly)
    local_compressions = local_stats.get("compressions", 0)
    local_saved = local_stats.get("total_tokens_saved", 0)
    if local_compressions > 0:
        lines.append(f"MCP Tool: {local_compressions} compressions, {local_saved:,} tokens saved")
        lines.append("")

    # Tip
    tip = summary.get("tip")
    if tip:
        lines.append(f"Tip: {tip}")

    return "\n".join(lines)


# Session-scoped TTL: content persists for the session (1 hour), not 5 minutes.
# The MCP server process lives as long as the coding session.
MCP_SESSION_TTL = 3600

# Shared stats file: all MCP instances (main + sub-agents) append here.
# headroom_stats aggregates across all instances within the session window.
# Respects HEADROOM_WORKSPACE_DIR.
SHARED_STATS_DIR = _paths.workspace_dir()
SHARED_STATS_FILE = _paths.session_stats_path()
SESSION_WINDOW_SECONDS = 7200  # 2 hours — events older than this are pruned


def _append_shared_event(event: dict[str, Any]) -> None:
    """Append an event to the shared stats file (cross-process, file-locked)."""
    try:
        SHARED_STATS_DIR.mkdir(parents=True, exist_ok=True)
        event["pid"] = os.getpid()
        line = json.dumps(event, separators=(",", ":")) + "\n"
        with open(SHARED_STATS_FILE, "a") as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        pass  # Never break compression because of stats


def _read_shared_events(window_seconds: int = SESSION_WINDOW_SECONDS) -> list[dict[str, Any]]:
    """Read shared events within the session time window, pruning old entries."""
    if not SHARED_STATS_FILE.exists():
        return []
    cutoff = time.time() - window_seconds
    events: list[dict[str, Any]] = []
    keep_lines: list[str] = []
    try:
        with open(SHARED_STATS_FILE) as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_SH)
            lines = f.readlines()
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_UN)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                if evt.get("timestamp", 0) >= cutoff:
                    events.append(evt)
                    keep_lines.append(line + "\n")
            except json.JSONDecodeError:
                continue
        # Prune old entries (only if we dropped some)
        if len(keep_lines) < len(lines):
            try:
                with open(SHARED_STATS_FILE, "w") as f:
                    if _HAS_FCNTL:
                        fcntl.flock(f, fcntl.LOCK_EX)
                    f.writelines(keep_lines)
                    if _HAS_FCNTL:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except Exception:
                pass
    except Exception:
        pass
    return events


@dataclass
class SessionStats:
    """Track compression statistics for the current MCP session."""

    compressions: int = 0
    retrievals: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens_saved: int = 0
    started_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)

    def record_compression(
        self,
        input_tokens: int,
        output_tokens: int,
        strategy: str,
    ) -> None:
        self.compressions += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens_saved += max(0, input_tokens - output_tokens)
        event = {
            "type": "compress",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "savings_percent": round((1 - output_tokens / input_tokens) * 100, 1)
            if input_tokens > 0
            else 0,
            "strategy": strategy,
            "timestamp": time.time(),
        }
        self.events.append(event)
        _append_shared_event(event)
        # Keep last 50 events
        if len(self.events) > 50:
            self.events = self.events[-50:]

    def record_retrieval(self, hash_key: str) -> None:
        self.retrievals += 1
        event = {
            "type": "retrieve",
            "hash": hash_key[:12],
            "timestamp": time.time(),
        }
        self.events.append(event)
        _append_shared_event(event)
        if len(self.events) > 50:
            self.events = self.events[-50:]

    def to_dict(self) -> dict[str, Any]:
        savings_pct = (
            round((self.total_tokens_saved / self.total_input_tokens) * 100, 1)
            if self.total_input_tokens > 0
            else 0
        )
        # Rough cost estimate (blended rate ~$3/1M input tokens)
        cost_saved = round(self.total_tokens_saved * 3.0 / 1_000_000, 4)

        return {
            "session_duration_seconds": round(time.time() - self.started_at),
            "compressions": self.compressions,
            "retrievals": self.retrievals,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "savings_percent": savings_pct,
            "estimated_cost_saved_usd": cost_saved,
            "recent_events": self.events[-10:],
        }


class HeadroomMCPServer:
    """MCP Server exposing Headroom's context engineering toolkit.

    Tools:
        headroom_compress — Compress content on demand. Stores original for
                           retrieval. Works without a proxy.
        headroom_retrieve — Retrieve original uncompressed content by hash.
                           Checks local store first, then proxy if configured.
        headroom_stats    — Session statistics: compressions, savings, cost.

    Modes:
        Standalone: Compression + retrieval happen locally. No proxy needed.
        With proxy: Retrieval also checks the proxy's compression store
                   (for content compressed by the proxy's automatic pipeline).
    """

    def __init__(
        self,
        proxy_url: str = DEFAULT_PROXY_URL,
        check_proxy: bool = True,
    ):
        self.proxy_url = proxy_url
        self.check_proxy = check_proxy
        self._http_client: httpx.AsyncClient | None = None  # type: ignore[assignment]
        self._stats = SessionStats()
        self._local_store: Any = None  # Lazy-initialized CompressionStore
        self._compressor_initialized = False
        # File read cache: path → (content_hash, ccr_hash, line_count, token_count)
        self._file_cache: dict[str, tuple[str, str, int, int]] = {}

        if not MCP_AVAILABLE or Server is None:
            raise ImportError("MCP SDK not installed. Install with: pip install mcp")

        self.server: Server = Server("headroom")
        self._setup_handlers()

    def _get_local_store(self) -> Any:
        """Get or create the local compression store (lazy init)."""
        if self._local_store is None:
            from headroom.cache.compression_store import CompressionStore

            self._local_store = CompressionStore(
                max_entries=500,
                default_ttl=MCP_SESSION_TTL,
            )
        return self._local_store

    def _compress_content(self, content: str) -> dict[str, Any]:
        """Compress content using Headroom's pipeline.

        Returns dict with compressed text, token counts, hash, etc.
        """
        from headroom.compress import compress

        # Wrap content as a tool message (most common compression target)
        messages = [{"role": "tool", "content": content}]

        result = compress(messages, model="claude-sonnet-4-5-20250929")

        compressed_content = result.messages[0].get("content", content)
        input_tokens = result.tokens_before
        output_tokens = result.tokens_after

        # Store original in local store for later retrieval
        store = self._get_local_store()
        hash_key = store.store(
            original=content,
            compressed=compressed_content
            if isinstance(compressed_content, str)
            else json.dumps(compressed_content),
            original_tokens=input_tokens,
            compressed_tokens=output_tokens,
            compression_strategy="mcp_compress",
            ttl=MCP_SESSION_TTL,
        )

        # Track stats
        strategy = (
            ", ".join(result.transforms_applied) if result.transforms_applied else "passthrough"
        )
        self._stats.record_compression(input_tokens, output_tokens, strategy)

        savings_pct = (
            round((1 - result.compression_ratio) * 100, 1) if result.compression_ratio < 1.0 else 0
        )

        return {
            "compressed": compressed_content,
            "hash": hash_key,
            "original_tokens": input_tokens,
            "compressed_tokens": output_tokens,
            "tokens_saved": max(0, input_tokens - output_tokens),
            "savings_percent": savings_pct,
            "transforms": result.transforms_applied,
            "note": f"Original stored with hash={hash_key}. Use mcp__headroom__{CCR_TOOL_NAME} to get full content later.",
        }

    async def _retrieve_content(
        self,
        hash_key: str,
        query: str | None,
    ) -> dict[str, Any]:
        """Retrieve content. Checks local store first, then proxy."""
        # Check local store first
        store = self._get_local_store()
        if query:
            results = store.search(hash_key, query)
            if results:
                self._stats.record_retrieval(hash_key)
                return {
                    "hash": hash_key,
                    "source": "local",
                    "query": query,
                    "results": results,
                    "count": len(results),
                }
        else:
            entry = store.retrieve(hash_key)
            if entry:
                self._stats.record_retrieval(hash_key)
                return {
                    "hash": hash_key,
                    "source": "local",
                    "original_content": entry.original_content,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                    "retrieval_count": entry.retrieval_count,
                }

        # Fall back to proxy if available
        if self.check_proxy and HTTPX_AVAILABLE:
            try:
                result = await self._retrieve_via_proxy(hash_key, query)
                if "error" not in result:
                    result["source"] = "proxy"
                    self._stats.record_retrieval(hash_key)
                    return result
            except Exception:
                pass  # Proxy unavailable, that's fine

        return {
            "error": "Content not found. It may have expired or the hash may be incorrect.",
            "hash": hash_key,
            "hint": "Content compressed via headroom_compress is stored for the session. "
            "Content compressed by the proxy has a shorter TTL (5 minutes).",
        }

    async def _retrieve_via_proxy(
        self,
        hash_key: str,
        query: str | None,
    ) -> dict[str, Any]:
        """Retrieve content via proxy's HTTP endpoint."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=15.0)

        url = f"{self.proxy_url}/v1/retrieve"
        payload: dict[str, str] = {"hash": hash_key}
        if query:
            payload["query"] = query

        response = await self._http_client.post(url, json=payload)

        if response.status_code == 404:
            return {"error": "Not found in proxy store", "hash": hash_key}

        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def _setup_handlers(self) -> None:
        """Register all MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            tools = [
                Tool(
                    name=COMPRESS_TOOL_NAME,
                    description=(
                        "Compress content to save context window space. "
                        "Use this on large tool outputs, file contents, search results, "
                        "or any content you want to shrink before reasoning over it. "
                        f"The original is stored and can be retrieved later via mcp__headroom__{CCR_TOOL_NAME}. "
                        "Returns compressed text + a hash for retrieval."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": (
                                    "The content to compress. Can be any text: file contents, "
                                    "JSON, search results, logs, code, etc."
                                ),
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name=CCR_TOOL_NAME,
                    description=(
                        "Retrieve original uncompressed content by hash. "
                        "Use this when you need full details from previously compressed content. "
                        "The hash comes from headroom_compress results or from compression "
                        "markers like [N items compressed... hash=abc123]."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hash": {
                                "type": "string",
                                "description": "Hash key from compression (e.g., 'abc123' from hash=abc123)",
                            },
                            "query": {
                                "type": "string",
                                "description": (
                                    "Optional search query to filter results. "
                                    "If provided, returns only items matching the query."
                                ),
                            },
                        },
                        "required": ["hash"],
                    },
                ),
                Tool(
                    name=STATS_TOOL_NAME,
                    description=(
                        "Show compression statistics for this session: "
                        "total compressions, tokens saved, estimated cost savings, "
                        "and recent compression events."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ]

            # Conditionally add headroom_read (behind feature flag)
            if _READ_ENABLED:
                tools.append(
                    Tool(
                        name=READ_TOOL_NAME,
                        description=(
                            "Read a file with smart caching. First read returns full content "
                            "and caches it. Subsequent reads of the same unchanged file return "
                            "a lightweight cache marker (~20 tokens instead of thousands). "
                            f"Use mcp__headroom__{CCR_TOOL_NAME} with the hash to get full content if needed. "
                            "Use this INSTEAD of the built-in Read tool for significant token savings."
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Absolute path to the file to read.",
                                },
                                "fresh": {
                                    "type": "boolean",
                                    "description": (
                                        "Force a fresh read, bypassing cache. Use after context "
                                        "compaction, in subagents, or when you need guaranteed "
                                        "current content."
                                    ),
                                },
                            },
                            "required": ["file_path"],
                        },
                    )
                )

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            try:
                if name == COMPRESS_TOOL_NAME:
                    return await self._handle_compress(arguments)
                elif name == CCR_TOOL_NAME:
                    return await self._handle_retrieve(arguments)
                elif name == STATS_TOOL_NAME:
                    return await self._handle_stats()
                elif name == READ_TOOL_NAME and _READ_ENABLED:
                    return await self._handle_read(arguments)
                else:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": f"Unknown tool: {name}"}),
                        )
                    ]
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(e)}),
                    )
                ]

    async def _handle_compress(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle headroom_compress tool call."""
        content = arguments.get("content")
        if not content:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "content parameter is required"}),
                )
            ]

        # Run compression in thread pool (it's CPU-bound)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._compress_content, content)

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_retrieve(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle headroom_retrieve tool call."""
        hash_key = arguments.get("hash")
        if not hash_key:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "hash parameter is required"}),
                )
            ]

        query = arguments.get("query")
        result = await self._retrieve_content(hash_key, query)

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_stats(self) -> list[TextContent]:
        """Handle headroom_stats tool call."""
        stats = self._stats.to_dict()

        # Add local store stats if available
        if self._local_store is not None:
            store_stats = self._local_store.get_stats()
            stats["store"] = {
                "entries": store_stats.get("entry_count", 0),
                "max_entries": store_stats.get("max_entries", 0),
            }

        # Aggregate cross-process stats (main session + sub-agents)
        my_pid = os.getpid()
        shared_events = _read_shared_events()
        other_events = [e for e in shared_events if e.get("pid") != my_pid]
        if other_events:
            other_compressions = [e for e in other_events if e.get("type") == "compress"]
            other_input = sum(e.get("input_tokens", 0) for e in other_compressions)
            other_output = sum(e.get("output_tokens", 0) for e in other_compressions)
            other_saved = max(0, other_input - other_output)
            stats["sub_agents"] = {
                "compressions": len(other_compressions),
                "retrievals": sum(1 for e in other_events if e.get("type") == "retrieve"),
                "tokens_saved": other_saved,
                "total_input_tokens": other_input,
                "total_output_tokens": other_output,
            }
            # Combined totals
            all_input = self._stats.total_input_tokens + other_input
            all_saved = self._stats.total_tokens_saved + other_saved
            stats["combined"] = {
                "total_compressions": self._stats.compressions + len(other_compressions),
                "total_tokens_saved": all_saved,
                "savings_percent": round(all_saved / all_input * 100, 1) if all_input > 0 else 0,
                "estimated_cost_saved_usd": round(all_saved * 3.0 / 1_000_000, 4),
            }

        # Fetch proxy stats and format summary if proxy is reachable
        if self.check_proxy and HTTPX_AVAILABLE:
            proxy_data = await self._fetch_full_proxy_stats()
            if proxy_data:
                summary = proxy_data.get("summary")
                if summary:
                    # Return clean formatted summary instead of raw JSON
                    formatted = _format_session_summary(summary, stats)
                    return [TextContent(type="text", text=formatted)]
                # Fallback: add proxy stats to local stats
                proxy_stats = self._extract_proxy_stats(proxy_data)
                if proxy_stats:
                    stats["proxy"] = proxy_stats

        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    async def _fetch_full_proxy_stats(self) -> dict[str, Any] | None:
        """Fetch full stats from the proxy (includes summary)."""
        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=15.0)
            response = await self._http_client.get(f"{self.proxy_url}/stats")
            if response.status_code != 200:
                return None
            result: dict[str, Any] = response.json()
            return result
        except Exception:
            return None

    @staticmethod
    def _extract_proxy_stats(data: dict[str, Any]) -> dict[str, Any] | None:
        """Extract key fields from full proxy stats (fallback when no summary)."""
        result: dict[str, Any] = {}
        if "requests_total" in data:
            result["requests_total"] = data["requests_total"]
        if "tokens_saved_total" in data:
            result["tokens_saved_total"] = data["tokens_saved_total"]
        cache = data.get("cache", data.get("caching", {}))
        if cache:
            result["cache"] = {
                "hits": cache.get("hits", cache.get("cache_hits", 0)),
                "misses": cache.get("misses", cache.get("cache_misses", 0)),
                "hit_rate": cache.get("hit_rate", cache.get("cache_hit_rate", 0)),
            }
        cost = data.get("cost", {})
        if cost:
            result["cost_saved_usd"] = cost.get("total_saved", cost.get("saved", 0))
        return result if result else None

    async def _handle_read(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle headroom_read tool call — file read with session caching."""
        import hashlib

        file_path = arguments.get("file_path", "")
        fresh = arguments.get("fresh", False)

        if not file_path:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path parameter is required"}),
                )
            ]

        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"File not found: {file_path}"}),
                )
            ]
        if not path.is_file():
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Not a file: {file_path}"}),
                )
            ]

        # Read file from disk. PR-A8 / P1-8: avoid lossy decode kwargs
        # in headroom/ccr/ — use the centralized safe-log decoder so
        # the project-wide grep stays clean (this path is for tool
        # output display, not SSE/wire path, so a replacement char on
        # invalid bytes is acceptable).
        try:
            from headroom.proxy.helpers import safe_decode_for_logging

            content = safe_decode_for_logging(path.read_bytes())
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Cannot read file: {e}"}),
                )
            ]

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:24]
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        str_path = str(path)

        # Check cache (unless fresh=true)
        if not fresh and str_path in self._file_cache:
            cached_hash, ccr_hash, cached_lines, cached_tokens = self._file_cache[str_path]
            if cached_hash == content_hash:
                # File unchanged — but is the CCR entry still alive?
                store = self._get_local_store()
                if store.exists(ccr_hash):
                    # CCR alive — return cache marker
                    self._stats.record_compression(cached_tokens, 5, "read_cache_hit")
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "status": "cached",
                                    "file": file_path,
                                    "lines": cached_lines,
                                    "unchanged": True,
                                    "hash": ccr_hash,
                                    "note": (
                                        f"File unchanged since first read ({cached_lines} lines, "
                                        f"~{cached_tokens} tokens). Content already in your context "
                                        f"from the first read. Call mcp__headroom__{CCR_TOOL_NAME}(hash='{ccr_hash}') "
                                        f"if you need the full content again."
                                    ),
                                },
                                indent=2,
                            ),
                        )
                    ]
                # CCR expired — clear stale cache, fall through to fresh read
                del self._file_cache[str_path]
            # File changed — fall through to fresh read

        # Fresh read: store in CCR and cache the hash
        store = self._get_local_store()
        ccr_hash = store.store(
            original=content,
            compressed=f"[File: {path.name}, {line_count} lines]",
            original_tokens=len(content.split()),
            compressed_tokens=5,
            tool_name="headroom_read",
            ttl=MCP_SESSION_TTL,
        )

        token_estimate = len(content.split())
        self._file_cache[str_path] = (content_hash, ccr_hash, line_count, token_estimate)

        # Return full content with line numbers (like Claude Code's Read tool)
        numbered_lines = []
        for i, line in enumerate(content.split("\n"), 1):
            numbered_lines.append(f"{i:>6}\t{line}")
        numbered_content = "\n".join(numbered_lines)

        return [
            TextContent(
                type="text",
                text=numbered_content,
            )
        ]

    async def run_stdio(self) -> None:
        """Run the server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            logger.info(f"Headroom MCP Server starting (proxy: {self.proxy_url})")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()


def create_ccr_mcp_server(
    proxy_url: str = DEFAULT_PROXY_URL,
    direct_mode: bool = False,
) -> HeadroomMCPServer:
    """Create a Headroom MCP server instance.

    Args:
        proxy_url: URL of the Headroom proxy server (for retrieval fallback).
        direct_mode: Ignored (kept for backward compatibility).

    Returns:
        HeadroomMCPServer instance.
    """
    return HeadroomMCPServer(proxy_url=proxy_url)


async def main() -> None:
    """Run the Headroom MCP server."""
    parser = argparse.ArgumentParser(
        description="Headroom MCP Server — Context engineering toolkit"
    )
    parser.add_argument(
        "--proxy-url",
        default=DEFAULT_PROXY_URL,
        help=f"Headroom proxy URL for retrieval fallback (default: {DEFAULT_PROXY_URL})",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="(Deprecated, ignored) Use direct CompressionStore access",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    server = HeadroomMCPServer(proxy_url=args.proxy_url)

    try:
        await server.run_stdio()
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
