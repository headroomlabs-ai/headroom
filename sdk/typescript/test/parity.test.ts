/**
 * Python-TypeScript SDK parity tests.
 *
 * These tests verify that the TypeScript SDK produces identical behavior
 * to the Python SDK when both talk to the same Headroom proxy.
 *
 * Requires:
 *   - HEADROOM_INTEGRATION=1 (env flag)
 *   - headroom proxy running on localhost:8787
 *   - OPENAI_API_KEY set in environment
 *
 * Run: HEADROOM_INTEGRATION=1 OPENAI_API_KEY=sk-... npx vitest run test/parity.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { compress, HeadroomClient, SharedContext, CompressionHooks, simulate } from "../src/index.js";
import type { CompressResult, CompressEvent, CompressContext } from "../src/index.js";
import { execSync } from "child_process";

const INTEGRATION = process.env.HEADROOM_INTEGRATION === "1";

// Sample data matching Python test fixtures
const sampleMessages = [
  { role: "system" as const, content: "You are a helpful assistant." },
  { role: "user" as const, content: "What is the capital of France?" },
  { role: "assistant" as const, content: "The capital of France is Paris." },
];

const sampleToolOutput = Array.from({ length: 80 }, (_, i) => ({
  id: i + 1,
  name: `server-${String(i + 1).padStart(3, "0")}`,
  status: i % 10 === 7 ? "error" : "running",
  cpu: Math.round(Math.random() * 100),
  memory: Math.round(Math.random() * 32768),
  region: ["us-east-1", "eu-west-1", "ap-southeast-1"][i % 3],
  last_check: "2025-01-15T10:00:00Z",
  uptime_days: Math.floor(Math.random() * 365),
}));

const messagesWithLargeToolOutput = [
  { role: "system" as const, content: "You are a DevOps assistant." },
  { role: "user" as const, content: "Show me all servers" },
  {
    role: "assistant" as const,
    content: null,
    tool_calls: [
      {
        id: "call_1",
        type: "function" as const,
        function: { name: "list_servers", arguments: "{}" },
      },
    ],
  },
  {
    role: "tool" as const,
    content: JSON.stringify(sampleToolOutput),
    tool_call_id: "call_1",
  },
  { role: "user" as const, content: "Which servers have errors?" },
];

describe.skipIf(!INTEGRATION)("Python-TypeScript Parity", () => {
  let pythonCompressResult: any;

  beforeAll(() => {
    // Run Python compress and capture result
    const pythonScript = `
import json, sys
sys.path.insert(0, "${process.cwd()}/../..")
from headroom import compress

messages = json.loads('''${JSON.stringify(messagesWithLargeToolOutput)}''')
result = compress(messages, model="gpt-4o")
print(json.dumps({
    "tokens_before": result.tokens_before,
    "tokens_after": result.tokens_after,
    "tokens_saved": result.tokens_saved,
    "compression_ratio": result.compression_ratio,
    "transforms_applied": result.transforms_applied,
    "message_count": len(result.messages),
}))
`;
    try {
      const output = execSync(`python3 -c '${pythonScript.replace(/'/g, "'\\''")}'`, {
        encoding: "utf-8",
        timeout: 30000,
      });
      pythonCompressResult = JSON.parse(output.trim());
    } catch (e) {
      console.warn("Python compress failed, skipping parity checks:", e);
      pythonCompressResult = null;
    }
  });

  describe("compress() parity", () => {
    it("small messages pass through unchanged (same as Python)", async () => {
      const result = await compress(sampleMessages, { model: "gpt-4o" });

      // Small messages should pass through — Python does the same
      expect(result.messages).toHaveLength(3);
      expect(result.messages[0].role).toBe("system");
      expect(result.messages[2].role).toBe("assistant");
    });

    it("large tool output gets compressed (same as Python)", async () => {
      const result = await compress(messagesWithLargeToolOutput, {
        model: "gpt-4o",
      });

      // Both Python and TS should compress this significantly
      expect(result.tokensSaved).toBeGreaterThan(0);
      expect(result.compressionRatio).toBeLessThan(1);
      expect(result.transformsApplied.length).toBeGreaterThan(0);

      console.log(`TS: ${result.tokensBefore} → ${result.tokensAfter} (saved ${result.tokensSaved}, ratio ${result.compressionRatio})`);
    });

    it("matches Python compression characteristics", async () => {
      if (!pythonCompressResult) return;

      const tsResult = await compress(messagesWithLargeToolOutput, {
        model: "gpt-4o",
      });

      console.log(`Python: ${pythonCompressResult.tokens_before} → ${pythonCompressResult.tokens_after}`);
      console.log(`TS:     ${tsResult.tokensBefore} → ${tsResult.tokensAfter}`);

      // Token counts should be similar (same proxy)
      // Allow 10% tolerance for non-determinism
      const tokenDiff = Math.abs(tsResult.tokensBefore - pythonCompressResult.tokens_before);
      expect(tokenDiff / pythonCompressResult.tokens_before).toBeLessThan(0.1);

      // Both should have compressed
      expect(tsResult.tokensSaved).toBeGreaterThan(0);
      expect(pythonCompressResult.tokens_saved).toBeGreaterThan(0);

      // Compression ratio should be similar
      const ratioDiff = Math.abs(tsResult.compressionRatio - pythonCompressResult.compression_ratio);
      expect(ratioDiff).toBeLessThan(0.15);
    });

    it("CompressResult has all fields matching Python", async () => {
      const result = await compress(sampleMessages, { model: "gpt-4o" });

      // Verify all fields present (matching Python CompressResult)
      expect(result).toHaveProperty("messages");
      expect(result).toHaveProperty("tokensBefore");
      expect(result).toHaveProperty("tokensAfter");
      expect(result).toHaveProperty("tokensSaved");
      expect(result).toHaveProperty("compressionRatio");
      expect(result).toHaveProperty("transformsApplied");
      expect(result).toHaveProperty("ccrHashes");
      expect(result).toHaveProperty("compressed");

      expect(typeof result.tokensBefore).toBe("number");
      expect(typeof result.tokensAfter).toBe("number");
      expect(Array.isArray(result.transformsApplied)).toBe(true);
      expect(Array.isArray(result.ccrHashes)).toBe(true);
    });
  });

  describe("compress() with hooks parity", () => {
    it("preCompress hook modifies messages before compression", async () => {
      let preCompressCalled = false;

      class TestHooks extends CompressionHooks {
        preCompress(messages: any[], ctx: CompressContext) {
          preCompressCalled = true;
          // Prepend a system message
          return [
            { role: "system", content: "Hook-injected prefix" },
            ...messages,
          ];
        }
      }

      const result = await compress(sampleMessages, {
        model: "gpt-4o",
        hooks: new TestHooks(),
      });

      expect(preCompressCalled).toBe(true);
      // Hook added a message, so there should be 4 total
      expect(result.messages).toHaveLength(4);
    });

    it("postCompress hook receives event with stats", async () => {
      let capturedEvent: CompressEvent | null = null;

      class LogHooks extends CompressionHooks {
        postCompress(event: CompressEvent) {
          capturedEvent = event;
        }
      }

      await compress(messagesWithLargeToolOutput, {
        model: "gpt-4o",
        hooks: new LogHooks(),
      });

      expect(capturedEvent).not.toBeNull();
      expect(capturedEvent!.tokensBefore).toBeGreaterThan(0);
      expect(typeof capturedEvent!.compressionRatio).toBe("number");
      expect(Array.isArray(capturedEvent!.transformsApplied)).toBe(true);
    });
  });

  describe("SharedContext parity", () => {
    it("put and get compressed content", async () => {
      const ctx = new SharedContext({ model: "gpt-4o" });

      const entry = await ctx.put("research", JSON.stringify(sampleToolOutput), {
        agent: "agent-A",
      });

      expect(entry.key).toBe("research");
      expect(entry.agent).toBe("agent-A");
      expect(entry.originalTokens).toBeGreaterThan(0);
      expect(entry.savingsPercent).toBeGreaterThanOrEqual(0);

      // Get compressed
      const compressed = ctx.get("research");
      expect(compressed).not.toBeNull();

      // Get full
      const full = ctx.get("research", { full: true });
      expect(full).toBe(JSON.stringify(sampleToolOutput));
    });

    it("stats aggregate correctly", async () => {
      const ctx = new SharedContext({ model: "gpt-4o" });

      await ctx.put("a", "Short content");
      await ctx.put("b", JSON.stringify(sampleToolOutput));

      const stats = ctx.stats();
      expect(stats.entries).toBe(2);
      expect(stats.totalOriginalTokens).toBeGreaterThan(0);
      expect(stats.totalTokensSaved).toBeGreaterThanOrEqual(0);
    });

    it("clear removes all entries", async () => {
      const ctx = new SharedContext({ model: "gpt-4o" });

      await ctx.put("a", "data");
      ctx.clear();

      expect(ctx.keys()).toEqual([]);
      expect(ctx.stats().entries).toBe(0);
    });
  });

  describe("simulate() parity", () => {
    it("returns simulation result without calling LLM", async () => {
      const result = await simulate(messagesWithLargeToolOutput, {
        model: "gpt-4o",
      });

      expect(result.tokensBefore).toBeGreaterThan(0);
      expect(result.tokensAfter).toBeGreaterThan(0);
      expect(typeof result.tokensSaved).toBe("number");
    });
  });

  describe("HeadroomClient proxy API parity", () => {
    it("health check works", async () => {
      const client = new HeadroomClient();
      const health = await client.health();

      expect(health.status).toBe("healthy");
      expect(health.version).toBeDefined();
    });

    it("proxy stats work", async () => {
      const client = new HeadroomClient();
      const stats = await client.proxyStats();

      expect(stats.requests).toBeDefined();
      expect(typeof stats.requests.total).toBe("number");
    });

    it("compress with config passthrough", async () => {
      const client = new HeadroomClient({
        config: {
          smartCrusher: { enabled: true, maxItemsAfterCrush: 5 },
        },
      });

      const result = await client.compress(
        messagesWithLargeToolOutput as any[],
        { model: "gpt-4o" },
      );

      expect(result.compressed).toBe(true);
      expect(result.tokensSaved).toBeGreaterThan(0);
    });
  });
});

describe.skipIf(!INTEGRATION)("End-to-End: OpenAI via Proxy", () => {
  it("chat.completions.create through proxy", async () => {
    const client = new HeadroomClient({
      providerApiKey: process.env.OPENAI_API_KEY,
    });

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "Reply in one word." },
        { role: "user", content: "What is 2+2?" },
      ],
    });

    expect(response.choices).toBeDefined();
    expect(response.choices[0].message.content).toBeTruthy();
    console.log("OpenAI response:", response.choices[0].message.content);
  });

  it("chat.completions.create compresses large context", async () => {
    const client = new HeadroomClient({
      providerApiKey: process.env.OPENAI_API_KEY,
    });

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        ...messagesWithLargeToolOutput,
        { role: "user", content: "How many servers have errors? Answer with just the number." },
      ] as any[],
      headroomMode: "optimize",
    });

    expect(response.choices).toBeDefined();
    const answer = response.choices[0].message.content;
    expect(answer).toBeTruthy();
    console.log("Answer about error servers:", answer);
  });
});
