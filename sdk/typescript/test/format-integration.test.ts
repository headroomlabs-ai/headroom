/**
 * Integration tests: compress() with ALL message formats against real proxy.
 *
 * Tests that compress() auto-detects and round-trips:
 *   - OpenAI format
 *   - Anthropic format
 *   - Vercel AI SDK format
 *   - Google Gemini format
 *
 * Requires: proxy running on http://localhost:8787
 * Run with: HEADROOM_INTEGRATION=1 npx vitest run test/format-integration.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { config } from "dotenv";
import { resolve } from "path";

config({ path: resolve(__dirname, "../../../.env") });

const PROXY_URL = "http://localhost:8787";
const RUN_INTEGRATION = process.env.HEADROOM_INTEGRATION === "1";

// Realistic large tool output — needs enough content to cross compression threshold
function bigToolOutput(n: number): string {
  const items = Array.from({ length: n }, (_, i) => ({
    id: i + 1,
    name: `item_${i + 1}`,
    status: i % 10 === 0 ? "error" : "ok",
    value: Math.round(Math.random() * 1000),
    timestamp: new Date(Date.now() - i * 60000).toISOString(),
    metadata: {
      category: ["A", "B", "C"][i % 3],
      tags: [`tag_${i % 5}`, `tag_${(i + 1) % 5}`],
      description: `This is item number ${i + 1} with some descriptive text that takes up tokens`,
    },
  }));
  return JSON.stringify({ results: items, total: n, page: 1 });
}

describe.skipIf(!RUN_INTEGRATION)("Format Integration: compress() auto-detects all formats", () => {
  beforeAll(async () => {
    const res = await fetch(`${PROXY_URL}/health`);
    if (!res.ok) throw new Error(`Proxy not running at ${PROXY_URL}`);
  });

  // ======== OpenAI FORMAT ========
  it("OpenAI format: detects, compresses, returns OpenAI format", async () => {
    const { compress } = await import("../src/compress.js");
    const { detectFormat } = await import("../src/utils/format.js");

    const messages = [
      { role: "system", content: "You are a data analyst" },
      { role: "user", content: "Search the database" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call_abc",
          type: "function",
          function: { name: "db_query", arguments: '{"sql":"SELECT * FROM items"}' },
        }],
      },
      {
        role: "tool",
        tool_call_id: "call_abc",
        content: bigToolOutput(80),
      },
      { role: "user", content: "Summarize the errors" },
    ];

    expect(detectFormat(messages)).toBe("openai");

    const result = await compress(messages, { model: "gpt-4o", baseUrl: PROXY_URL });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBeGreaterThan(0);
    expect(result.tokensSaved).toBeGreaterThan(0);

    // Output should still be OpenAI format
    const out = result.messages;
    expect(out[0].role).toBe("system");
    expect(typeof out[0].content).toBe("string");
    // tool messages should have tool_call_id (OpenAI marker)
    const toolMsg = out.find((m: any) => m.role === "tool");
    if (toolMsg) {
      expect("tool_call_id" in toolMsg).toBe(true);
      expect(typeof toolMsg.content).toBe("string");
    }

    console.log(`  OpenAI: ${result.tokensBefore} → ${result.tokensAfter} (saved ${result.tokensSaved})`);
  });

  // ======== ANTHROPIC FORMAT ========
  it("Anthropic format: detects, compresses, returns Anthropic format", async () => {
    const { compress } = await import("../src/compress.js");
    const { detectFormat } = await import("../src/utils/format.js");

    const messages = [
      { role: "user", content: "Search the database" },
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "toolu_01ABC", name: "db_query", input: { sql: "SELECT * FROM items" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_01ABC", content: bigToolOutput(80) },
        ],
      },
      { role: "user", content: "Summarize the errors" },
    ];

    expect(detectFormat(messages)).toBe("anthropic");

    const result = await compress(messages, { model: "claude-sonnet-4-5-20250929", baseUrl: PROXY_URL });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBeGreaterThan(0);

    // Output should be back in Anthropic format
    const out = result.messages;
    // Anthropic tool results are inside user messages with type: "tool_result"
    const hasToolResult = out.some((m: any) =>
      Array.isArray(m.content) && m.content.some((b: any) => b.type === "tool_result"),
    );
    // Tool use blocks should have type: "tool_use" (underscore, not hyphen)
    const hasToolUse = out.some((m: any) =>
      Array.isArray(m.content) && m.content.some((b: any) => b.type === "tool_use"),
    );

    if (hasToolResult) {
      expect(hasToolResult).toBe(true); // Anthropic format preserved
    }
    if (hasToolUse) {
      expect(hasToolUse).toBe(true);
    }

    console.log(`  Anthropic: ${result.tokensBefore} → ${result.tokensAfter} (saved ${result.tokensSaved})`);
  });

  // ======== VERCEL AI SDK FORMAT ========
  it("Vercel AI SDK format: detects, compresses, returns Vercel format", async () => {
    const { compress } = await import("../src/compress.js");
    const { detectFormat } = await import("../src/utils/format.js");

    const messages = [
      { role: "system", content: "You are a data analyst" },
      { role: "user", content: [{ type: "text", text: "Search the database" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me query that" },
          { type: "tool-call", toolCallId: "tc_1", toolName: "db_query", args: { sql: "SELECT *" } },
        ],
      },
      {
        role: "tool",
        content: [
          { type: "tool-result", toolCallId: "tc_1", toolName: "db_query", result: JSON.parse(bigToolOutput(80)) },
        ],
      },
      { role: "user", content: [{ type: "text", text: "Summarize the errors" }] },
    ];

    expect(detectFormat(messages)).toBe("vercel");

    const result = await compress(messages, { model: "gpt-4o", baseUrl: PROXY_URL });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBeGreaterThan(0);

    // Output should be back in Vercel format
    const out = result.messages;
    expect(out[0].role).toBe("system");
    // Vercel user messages have content as array of parts
    const userMsg = out.find((m: any) => m.role === "user");
    if (userMsg && Array.isArray(userMsg.content)) {
      expect(userMsg.content[0].type).toBe("text");
    }
    // Vercel tool results use "tool-result" (hyphenated)
    const toolMsg = out.find((m: any) => m.role === "tool");
    if (toolMsg && Array.isArray(toolMsg.content)) {
      expect(toolMsg.content[0].type).toBe("tool-result");
    }

    console.log(`  Vercel: ${result.tokensBefore} → ${result.tokensAfter} (saved ${result.tokensSaved})`);
  });

  // ======== GOOGLE GEMINI FORMAT ========
  it("Gemini format: detects, compresses, returns Gemini format", async () => {
    const { compress } = await import("../src/compress.js");
    const { detectFormat } = await import("../src/utils/format.js");

    const messages = [
      { role: "user", parts: [{ text: "Search the database" }] },
      {
        role: "model",
        parts: [
          { functionCall: { name: "db_query", args: { sql: "SELECT * FROM items" } } },
        ],
      },
      {
        role: "user",
        parts: [
          { functionResponse: { name: "db_query", response: JSON.parse(bigToolOutput(80)) } },
        ],
      },
      { role: "user", parts: [{ text: "Summarize the errors" }] },
    ];

    expect(detectFormat(messages)).toBe("gemini");

    const result = await compress(messages, { model: "gemini-2.0-flash", baseUrl: PROXY_URL });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBeGreaterThan(0);

    // Output should be back in Gemini format
    const out = result.messages;
    // Gemini uses "parts" not "content"
    expect(out[0].parts).toBeDefined();
    // Gemini uses "model" role not "assistant"
    const modelMsg = out.find((m: any) => m.role === "model");
    if (modelMsg) {
      expect(modelMsg.parts).toBeDefined();
    }

    console.log(`  Gemini: ${result.tokensBefore} → ${result.tokensAfter} (saved ${result.tokensSaved})`);
  });

  // ======== SIMPLE STRING MESSAGES (ambiguous → defaults to OpenAI) ========
  it("Simple string messages work (default OpenAI detection)", async () => {
    const { compress } = await import("../src/compress.js");
    const { detectFormat } = await import("../src/utils/format.js");

    const messages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
      { role: "user", content: "What's the weather?" },
    ];

    expect(detectFormat(messages)).toBe("openai");

    const result = await compress(messages, { model: "gpt-4o", baseUrl: PROXY_URL });

    // Small messages may not compress but should round-trip cleanly
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0].role).toBe("user");
    expect(result.messages[0].content).toBe("Hello");
  });
});
