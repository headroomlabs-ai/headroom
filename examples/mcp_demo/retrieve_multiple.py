"""Compress two outputs and retrieve both through the real MCP stdio server.

Run from the repository root (requires headroom-ai[mcp], no provider key):
    HEADROOM_OFFLINE=1 python -m examples.mcp_demo.retrieve_multiple
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    originals = [
        json.dumps([{"id": i, "status": "ok"} for i in range(80)]),
        "\n".join(f"INFO task {i} completed" for i in range(120)) + "\nFATAL disk full",
    ]
    # Keep this demonstration's cache and session statistics separate.
    with tempfile.TemporaryDirectory(prefix="headroom-batch-demo-") as workspace:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "headroom.cli", "mcp", "serve"],
            env={**os.environ, "HEADROOM_WORKSPACE_DIR": workspace, "HEADROOM_BEACON": "off"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                hashes = []
                for original in originals:
                    result = await session.call_tool("headroom_compress", {"content": original})
                    assert result.content[0].type == "text"
                    payload = json.loads(result.content[0].text)
                    assert payload.get("hash"), payload
                    hashes.append(payload["hash"])

                result = await session.call_tool(
                    "headroom_retrieve_multiple",
                    {"hashes": [hashes[1], "missing", hashes[0], hashes[1]]},
                )
                assert result.content[0].type == "text"
                results = json.loads(result.content[0].text)["results"]
                assert [r["hash"] for r in results] == [hashes[1], "missing", hashes[0], hashes[1]]
                assert results[0]["original_content"] == originals[1]
                assert "error" in results[1]
                assert results[2]["original_content"] == originals[0]
                assert results[3]["original_content"] == originals[1]
                print(
                    "MCP stdio: 4 ordered results; 3 full originals verified; missing hash reported."
                )


if __name__ == "__main__":
    asyncio.run(main())
