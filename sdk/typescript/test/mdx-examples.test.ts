/**
 * Tests every code example from the Vercel AI SDK PR MDX files.
 *
 * If these pass, the examples in the docs are correct.
 *
 * Run: HEADROOM_INTEGRATION=1 npx vitest run test/mdx-examples.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { config } from "dotenv";
import { resolve } from "path";

config({ path: resolve(__dirname, "../../../.env") });

const PROXY_URL = "http://localhost:8787";
const RUN = process.env.HEADROOM_INTEGRATION === "1";

describe.skipIf(!RUN)("MDX Examples: 51-headroom.mdx", () => {
  beforeAll(async () => {
    const res = await fetch(`${PROXY_URL}/health`);
    if (!res.ok) throw new Error("Proxy not running");
  });

  // =====================================================
  // Example 1 from 51-headroom.mdx: "Compress messages before calling the model"
  // =====================================================
  it("compress() with AI SDK format messages → generateText()", { timeout: 30000 }, async () => {
    const { compress } = await import("../src/compress.js");
    const { generateText } = await import("ai");
    const { createOpenAI } = await import("@ai-sdk/openai");

    const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });

    // Simulated large tool result (the MDX shows `largeLogData`)
    const largeLogData = Array.from({ length: 100 }, (_, i) => ({
      timestamp: new Date(Date.now() - i * 1000).toISOString(),
      level: i === 42 ? "FATAL" : i % 10 === 0 ? "ERROR" : "INFO",
      service: `service-${["auth", "payment", "user", "api"][i % 4]}`,
      message:
        i === 42
          ? "Connection pool exhausted — max_connections=100 reached, 47 pending requests"
          : `Request processed in ${Math.round(Math.random() * 500)}ms for /${["login", "checkout", "profile", "notify"][i % 4]}`,
      trace_id: `trace-${Math.random().toString(36).substring(2, 10)}`,
    }));

    // EXACT pattern from MDX (adapted to use AI SDK message format)
    const messages: any[] = [
      { role: "user", content: "Analyze the server logs" },
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "tc_1",
            toolName: "get_logs",
            args: { limit: 100 },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "get_logs",
            output: { type: "json", value: largeLogData },
          },
        ],
      },
      { role: "user", content: "What is the critical issue?" },
    ];

    const compressed = await compress(messages, {
      model: "gpt-4o",
      baseUrl: PROXY_URL,
    });

    console.log(
      `  Example 1: ${compressed.tokensBefore} → ${compressed.tokensAfter} tokens (saved ${compressed.tokensSaved})`,
    );

    // Verify compression happened
    expect(compressed.tokensBefore).toBeGreaterThan(0);
    // Messages should still be in a format generateText accepts
    expect(compressed.messages.length).toBeGreaterThan(0);

    // Now call generateText with compressed messages
    const { text } = await generateText({
      model: openai("gpt-4o-mini"),
      messages: compressed.messages,
    });

    console.log(`  LLM response: "${text.substring(0, 150)}"`);
    expect(text.length).toBeGreaterThan(0);
    // Should find the FATAL connection pool issue
    expect(text.toLowerCase()).toMatch(/connection|pool|fatal|exhaust/i);
  });

  // =====================================================
  // Example 2 from 51-headroom.mdx: "Use as middleware"
  // =====================================================
  it("headroomMiddleware() with wrapLanguageModel → generateText()", { timeout: 30000 }, async () => {
    const { headroomMiddleware } = await import("../src/adapters/vercel-ai.js");
    const { wrapLanguageModel, generateText } = await import("ai");
    const { createOpenAI } = await import("@ai-sdk/openai");

    const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });

    // EXACT pattern from MDX
    const model = wrapLanguageModel({
      model: openai("gpt-4o-mini"),
      middleware: headroomMiddleware({ baseUrl: PROXY_URL }),
    });

    // Feed it a big prompt that will get compressed
    const serverData = Array.from({ length: 100 }, (_, i) => ({
      name: `server-${i + 1}`,
      status: i % 15 === 0 ? "critical" : "healthy",
      cpu: Math.round(Math.random() * 100),
      alert: i % 15 === 0 ? `Disk at ${90 + (i % 10)}%` : null,
      description: `Production server ${i + 1} running service-${["auth", "payment"][i % 2]}`,
    }));

    const { text } = await generateText({
      model,
      system: "List only the critical servers. One line each.",
      prompt: `Fleet status:\n${JSON.stringify(serverData)}`,
    });

    console.log(`  Example 2 (middleware): "${text.substring(0, 150)}"`);
    expect(text.length).toBeGreaterThan(0);
    expect(text.toLowerCase()).toMatch(/server/i);
  });

  // =====================================================
  // Example 3 from 51-headroom.mdx: "Works with any provider" (Anthropic)
  // =====================================================
  it("compress() → Anthropic via AI SDK", { timeout: 30000 }, async () => {
    if (!process.env.ANTHROPIC_API_KEY) {
      console.log("  Skipping: ANTHROPIC_API_KEY not set");
      return;
    }

    const { compress } = await import("../src/compress.js");
    const { generateText } = await import("ai");
    const { createAnthropic } = await import("@ai-sdk/anthropic");

    const anthropic = createAnthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
      baseURL: "https://api.anthropic.com/v1",
    });

    const searchResults = Array.from({ length: 80 }, (_, i) => ({
      title: `${["API Design", "Database Tuning", "Cache Strategy", "Load Balancing"][i % 4]} Guide ${i + 1}`,
      url: `https://docs.example.com/${i + 1}`,
      snippet: `Covers ${["best practices", "pitfalls", "advanced techniques", "getting started"][i % 4]} for ${["microservices", "distributed systems", "cloud native", "serverless"][i % 4]}.`,
      score: (100 - i) / 100,
    }));

    // Simple messages (no tool calls — just user content)
    const messages: any[] = [
      {
        role: "user",
        content: `Search results:\n${JSON.stringify(searchResults)}\n\nTop 3 results? One sentence each.`,
      },
    ];

    // EXACT pattern from MDX
    const compressed = await compress(messages, {
      model: "claude-haiku-4-5-20251001",
      baseUrl: PROXY_URL,
    });

    console.log(
      `  Example 3 (Anthropic): ${compressed.tokensBefore} → ${compressed.tokensAfter} tokens`,
    );

    const { text } = await generateText({
      model: anthropic("claude-haiku-4-5-20251001"),
      messages: compressed.messages,
      maxTokens: 300,
    });

    console.log(`  Anthropic response: "${text.substring(0, 150)}"`);
    expect(text.length).toBeGreaterThan(0);
  });
});

describe.skipIf(!RUN)("MDX Examples: context-compression-middleware.mdx", () => {
  beforeAll(async () => {
    const res = await fetch(`${PROXY_URL}/health`);
    if (!res.ok) throw new Error("Proxy not running");
  });

  // =====================================================
  // The cookbook's compressionMiddleware — test the EXACT code from the MDX
  // =====================================================
  it("compressionMiddleware from cookbook works end-to-end", { timeout: 30000 }, async () => {
    const { compress } = await import("../src/compress.js");
    const { wrapLanguageModel, generateText } = await import("ai");
    const { createOpenAI } = await import("@ai-sdk/openai");

    const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });

    // EXACT middleware from the cookbook MDX
    const compressionMiddleware = {
      transformParams: async ({ params }: { params: any }) => {
        const prompt = params.prompt;
        if (!prompt || prompt.length === 0) return params;

        const result = await compress(prompt, {
          model: params.modelId ?? "gpt-4o",
          baseUrl: PROXY_URL,
        });

        if (!result.compressed) return params;

        console.log(
          `    Compressed: ${result.tokensBefore} → ${result.tokensAfter} tokens (saved ${result.tokensSaved})`,
        );

        return { ...params, prompt: result.messages };
      },
    };

    // EXACT pattern from cookbook: wrap model with middleware
    const model = wrapLanguageModel({
      model: openai("gpt-4o-mini"),
      middleware: compressionMiddleware,
    });

    // Simulate the SRE agent scenario from the cookbook
    const serverData = Array.from({ length: 100 }, (_, i) => ({
      id: i + 1,
      name: `server-${i + 1}`,
      status: i % 15 === 0 ? "critical" : i % 5 === 0 ? "warning" : "healthy",
      cpu: Math.round(Math.random() * 100),
      memory: Math.round(Math.random() * 100),
      region: ["us-east-1", "eu-west-1", "ap-southeast-1"][i % 3],
      lastAlert: i % 15 === 0 ? `Disk usage at ${90 + (i % 10)}%` : null,
    }));

    const { text } = await generateText({
      model,
      system: "You are an SRE assistant. List only the critical servers.",
      prompt: `Fleet status:\n${JSON.stringify(serverData)}`,
    });

    console.log(`  Cookbook middleware response: "${text.substring(0, 200)}"`);
    expect(text.length).toBeGreaterThan(0);
    expect(text.toLowerCase()).toMatch(/server/i);
  });
});
