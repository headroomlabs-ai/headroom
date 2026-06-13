"""Headroom CCR retrieve plugin.

The headroom proxy (127.0.0.1:8787) compresses large tool outputs in LLM
requests, replacing them with markers like ``[N items compressed ...
hash=abc123]`` or ``<<ccr:abc123>>``. This plugin gives Hermes a tool to fetch the original
uncompressed content back from the proxy's compression store, so compressed
markers are no longer a black box.

Storage is in-memory on the proxy side with a TTL — expired or
post-proxy-restart hashes return 404 and the tool reports that clearly.
"""

from __future__ import annotations

import httpx
from tools.registry import tool_error, tool_result

_PROXY_URL = "http://127.0.0.1:8787"

HEADROOM_RETRIEVE_SCHEMA = {
    "name": "headroom_retrieve",
    "description": (
        "Retrieve the original uncompressed content behind a headroom "
        "compression marker. Markers look like "
        "'[N items compressed ... hash=abc123]' OR '<<ccr:abc123>>' OR "
        "'<<ccr:abc123,base64,4.5KB>>'. They are NOT file paths — never try "
        "to cat/read them. When you see one in a tool result or in "
        "conversation history, call this tool with the hash (the hex string "
        "after 'hash=' or 'ccr:') to read the full original content instead "
        "of guessing or re-running the command. For very large results, pass "
        "the optional 'query' to filter to the relevant parts (BM25 search). "
        "Content expires after a TTL — if expired, re-run the original "
        "command instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hash": {
                "type": "string",
                "description": "Hash from the compression marker, e.g. 'abc123' from '[... hash=abc123]' or '<<ccr:abc123>>'",
            },
            "query": {
                "type": "string",
                "description": "Optional search query to filter large results to relevant items",
            },
        },
        "required": ["hash"],
    },
}


def _handle_headroom_retrieve(args: dict, **kw) -> str:
    hash_key = str(args.get("hash") or "").strip()
    # Tolerate the model passing the whole marker instead of the bare hash:
    # '<<ccr:abc123,base64,4.5KB>>' / 'ccr:abc123' / 'hash=abc123' -> 'abc123'
    hash_key = hash_key.strip("<>").removeprefix("ccr:").removeprefix("hash=")
    hash_key = hash_key.split(",")[0].strip()
    if not hash_key:
        return tool_error(
            "hash is required (from a '[... hash=abc123]' or '<<ccr:abc123>>' marker)"
        )

    payload: dict = {"hash": hash_key}
    query = str(args.get("query") or "").strip()
    if query:
        payload["query"] = query

    try:
        resp = httpx.post(f"{_PROXY_URL}/v1/retrieve", json=payload, timeout=15)
    except httpx.HTTPError as exc:
        return tool_error(
            f"headroom proxy unreachable at {_PROXY_URL} ({type(exc).__name__}). "
            "The proxy may be down; re-run the original command to get the data."
        )

    if resp.status_code == 404:
        return tool_error(
            "Content not found: expired (TTL passed) or proxy restarted. "
            "Re-run the original command to regenerate the data."
        )
    if resp.status_code != 200:
        return tool_error(f"headroom proxy returned HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    return tool_result(
        {
            "original_content": data.get("original_content", ""),
            "original_tokens": data.get("original_tokens"),
            "tool_name": data.get("tool_name"),
        }
    )


def register(ctx) -> None:
    """Register the headroom_retrieve tool. Called by the plugin loader."""
    ctx.register_tool(
        name="headroom_retrieve",
        toolset="headroom",
        schema=HEADROOM_RETRIEVE_SCHEMA,
        handler=_handle_headroom_retrieve,
        emoji="🗜️",
    )
