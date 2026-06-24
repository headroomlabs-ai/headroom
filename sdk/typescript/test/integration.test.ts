/**
 * Integration tests for the Headroom TypeScript SDK.
 *
 * These tests run against a real Headroom proxy server.
 * They require:
 *   - The proxy running on http://localhost:8787
 *   - OPENAI_API_KEY and ANTHROPIC_API_KEY in .env
 *
 * Run with: HEADROOM_INTEGRATION=1 npx vitest run test/integration.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { config } from "dotenv";
import { resolve } from "path";

// Load .env from the project root
config({ path: resolve(__dirname, "../../../.env") });

const PROXY_URL = "http://localhost:8787";
const RUN_INTEGRATION = process.env.HEADROOM_INTEGRATION === "1";

// Large tool output that should get meaningfully compressed
function makeLargeToolOutput(itemCount: number): string {
  const items = Array.from({ length: itemCount }, (_, i) => ({
    id: i + 1,
    name: `item_${i + 1}`,
    status: i % 10 === 0 ? "error" : "ok",
    value: Math.random() * 1000,
    timestamp: new Date(Date.now() - i * 60000).toISOString(),
    metadata: {
      category: ["A", "B", "C"][i % 3],
      tags: [`tag_${i % 5}`, `tag_${(i + 1) % 5}`],
      description: `This is item number ${i + 1} with some descriptive text that takes up tokens`,
    },
  }));
  return JSON.stringify({ results: items, total: itemCount, page: 1 });
}

describe.skipIf(!RUN_INTEGRATION)("Integration: compress() with real proxy", () => {
  beforeAll(async () => {
    // Verify proxy is running
    try {
      const res = await fetch(`${PROXY_URL}/health`);
      if (!res.ok) throw new Error(`Proxy health check failed: ${res.status}`);
    } catch (e) {
      throw new Error(
        `Proxy not running at ${PROXY_URL}. Start with: headroom proxy --port 8787\n${e}`,
      );
    }
  });

  it("compress() returns compressed messages with real proxy", async () => {
    const { compress } = await import("../src/compress.js");

    const messages = [
      { role: "user" as const, content: "Search for recent data" },
      {
        role: "assistant" as const,
        content: null,
        tool_calls: [
          {
            id: "tc_1",
            type: "function" as const,
            function: {
              name: "search_database",
              arguments: '{"query": "recent data", "limit": 100}',
            },
          },
        ],
      },
      {
        role: "tool" as const,
        content: makeLargeToolOutput(100),
        tool_call_id: "tc_1",
      },
      { role: "user" as const, content: "Summarize the key findings" },
    ];

    const result = await compress(messages, {
      model: "gpt-4o",
      baseUrl: PROXY_URL,
    });

    // Verify structure
    expect(result.messages).toBeDefined();
    expect(Array.isArray(result.messages)).toBe(true);
    expect(result.messages.length).toBeGreaterThan(0);
    expect(result.tokensBefore).toBeGreaterThan(0);
    expect(result.tokensAfter).toBeGreaterThan(0);
    expect(result.tokensSaved).toBeGreaterThanOrEqual(0);
    expect(result.compressionRatio).toBeGreaterThan(0);
    expect(result.compressionRatio).toBeLessThanOrEqual(1);
    expect(result.compressed).toBe(true);

    // With 100 items of tool output, we expect meaningful compression
    console.log(
      `  compress(): ${result.tokensBefore} → ${result.tokensAfter} tokens ` +
        `(saved ${result.tokensSaved}, ratio ${result.compressionRatio.toFixed(2)})`,
    );
    console.log(`  transforms: ${result.transformsApplied.join(", ")}`);

    // Should save at least some tokens with 100 items
    expect(result.tokensSaved).toBeGreaterThan(0);
  });

  it("compress() preserves message structure for small inputs", async () => {
    const { compress } = await import("../src/compress.js");

    const messages = [
      { role: "system" as const, content: "You are helpful" },
      { role: "user" as const, content: "Hello" },
    ];

    const result = await compress(messages, {
      model: "gpt-4o",
      baseUrl: PROXY_URL,
    });

    // Small messages should pass through with structure intact
    expect(result.messages.length).toBe(2);
    expect(result.messages[0].role).toBe("system");
    expect(result.messages[1].role).toBe("user");
  });

  it("compress() works with claude model name", async () => {
    const { compress } = await import("../src/compress.js");

    const messages = [
      { role: "user" as const, content: "Search results" },
      {
        role: "tool" as const,
        content: makeLargeToolOutput(50),
        tool_call_id: "tc_1",
      },
    ];

    const result = await compress(messages, {
      model: "claude-sonnet-4-5-20250929",
      baseUrl: PROXY_URL,
    });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBeGreaterThan(0);
    console.log(
      `  claude model: ${result.tokensBefore} → ${result.tokensAfter} tokens`,
    );
  });

  it("HeadroomClient can be reused across calls", async () => {
    const { HeadroomClient } = await import("../src/client.js");

    const client = new HeadroomClient({ baseUrl: PROXY_URL });

    const result1 = await client.compress(
      [
        { role: "user", content: "First call" },
        {
          role: "tool",
          content: makeLargeToolOutput(50),
          tool_call_id: "tc_1",
        },
      ],
      { model: "gpt-4o" },
    );

    const result2 = await client.compress(
      [
        { role: "user", content: "Second call" },
        {
          role: "tool",
          content: makeLargeToolOutput(50),
          tool_call_id: "tc_2",
        },
      ],
      { model: "gpt-4o" },
    );

    expect(result1.compressed).toBe(true);
    expect(result2.compressed).toBe(true);
  });

  it("compress() with empty messages returns empty", async () => {
    const { compress } = await import("../src/compress.js");

    const result = await compress([], {
      model: "gpt-4o",
      baseUrl: PROXY_URL,
    });

    expect(result.messages).toEqual([]);
    expect(result.tokensSaved).toBe(0);
  });

  it("Vercel AI SDK adapter converts and compresses", async () => {
    const { compressVercelMessages } = await import(
      "../src/adapters/vercel-ai.js"
    );

    const vercelMessages = [
      { role: "system", content: "Be helpful" },
      {
        role: "user",
        content: [{ type: "text", text: "Analyze this data" }],
      },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me search" },
          {
            type: "tool-call",
            toolCallId: "tc_1",
            toolName: "search",
            args: { query: "data" },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "search",
            result: JSON.parse(makeLargeToolOutput(80)),
          },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "Summarize" }],
      },
    ];

    const result = await compressVercelMessages(vercelMessages, {
      model: "gpt-4o",
      baseUrl: PROXY_URL,
    });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBeGreaterThan(0);

    // Messages should be back in Vercel format
    expect(result.messages[0].role).toBe("system");

    console.log(
      `  Vercel adapter: ${result.tokensBefore} → ${result.tokensAfter} tokens ` +
        `(saved ${result.tokensSaved})`,
    );
  });
});
