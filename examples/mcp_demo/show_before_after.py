"""Show BEFORE/AFTER code changes for MCP integration.

This demonstrates the minimal code change needed to add Headroom
compression to MCP tool outputs in your host application.

Run with:
    PYTHONPATH=. python -m examples.mcp_demo.show_before_after
"""


def main():
    print("\n" + "=" * 70)
    print("HEADROOM MCP INTEGRATION - DEVELOPER EXPERIENCE")
    print("=" * 70)

    # =========================================================================
    # Option 1: Standalone Function (Simplest)
    # =========================================================================
    print("\n" + "─" * 70)
    print("OPTION 1: Standalone Function (2 lines to add)")
    print("─" * 70)

    print("\nBEFORE (in your MCP host application):")
    print("-" * 40)
    before_standalone = """
# Your MCP host application
result = await mcp_client.call_tool("search_logs", {"service": "api"})
messages.append({"role": "tool", "content": result})
"""
    print(before_standalone)

    print("\nAFTER (with Headroom compression):")
    print("-" * 40)
    after_standalone = """
from headroom.integrations.mcp import compress_tool_result  # ADD THIS

# Your MCP host application
result = await mcp_client.call_tool("search_logs", {"service": "api"})
compressed = compress_tool_result(                          # ADD THIS
    content=result,                                         # ADD THIS
    tool_name="search_logs",                                # ADD THIS
    user_query="find errors in api",                        # ADD THIS
)                                                           # ADD THIS
messages.append({"role": "tool", "content": compressed})
"""
    print(after_standalone)

    # =========================================================================
    # Option 2: Client Wrapper (Zero-Touch After Setup)
    # =========================================================================
    print("\n" + "─" * 70)
    print("OPTION 2: Client Wrapper (wrap once, forget)")
    print("─" * 70)

    print("\nBEFORE:")
    print("-" * 40)
    before_wrapper = """
from mcp import Client

# Create MCP client
client = Client(transport)

# Use client normally
result = await client.call_tool("search_logs", {"service": "api"})
"""
    print(before_wrapper)

    print("\nAFTER:")
    print("-" * 40)
    after_wrapper = """
from mcp import Client
from headroom.integrations.mcp import HeadroomMCPClientWrapper  # ADD THIS

# Create MCP client
base_client = Client(transport)
client = HeadroomMCPClientWrapper(base_client)  # WRAP IT (1 line)

# Use client normally - compression is automatic!
result = await client.call_tool("search_logs", {"service": "api"})
"""
    print(after_wrapper)

    # =========================================================================
    # Option 3: With Metrics
    # =========================================================================
    print("\n" + "─" * 70)
    print("OPTION 3: With Metrics (track savings)")
    print("─" * 70)

    print("\nCode with metrics tracking:")
    print("-" * 40)
    with_metrics = """
from headroom.integrations.mcp import compress_tool_result_with_metrics

result = await mcp_client.call_tool("search_logs", {"service": "api"})
compression = compress_tool_result_with_metrics(
    content=result,
    tool_name="search_logs",
    user_query="find errors",
)

print(f"Tokens saved: {compression.tokens_saved}")
print(f"Compression: {compression.compression_ratio:.1%}")
print(f"Errors preserved: {compression.errors_preserved}")

messages.append({"role": "tool", "content": compression.compressed_content})
"""
    print(with_metrics)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: What Developers Need to Do")
    print("=" * 70)

    print("""
1. SIMPLEST (Standalone Function):
   - Add 1 import
   - Wrap tool result with compress_tool_result()
   - 2 lines of code change

2. EASIEST (Client Wrapper):
   - Add 1 import
   - Wrap your MCP client once
   - All subsequent tool calls automatically compressed

3. OBSERVABILITY (With Metrics):
   - Use compress_tool_result_with_metrics()
   - Get full metrics: tokens_saved, compression_ratio, errors_preserved
   - Track savings over time

Key Benefits:
- 70-85% token reduction on large tool outputs
- 100% ERROR preservation (log entries, exceptions, failures)
- Zero config needed (smart defaults for common MCP servers)
- Full schema preservation (JSON structure intact)
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
