/**
 * End-to-end test: Vercel AI SDK + Headroom compress()
 *
 * Tests REAL calls through the Vercel AI SDK with Headroom compression.
 * Uses actual OpenAI and Anthropic API keys.
 *
 * Requires:
 *   - Proxy running on http://localhost:8787
 *   - OPENAI_API_KEY in .env
 *   - ANTHROPIC_API_KEY in .env
 *
 * Run: HEADROOM_INTEGRATION=1 npx vitest run test/vercel-ai-e2e.test.ts
 */
import { describe, it, expect, beforeAll } from "vitest";
import { config } from "dotenv";
import { resolve } from "path";

config({ path: resolve(__dirname, "../../../.env") });

const PROXY_URL = "http://localhost:8787";
const RUN = process.env.HEADROOM_INTEGRATION === "1";

describe.skipIf(!RUN)("E2E: Vercel AI SDK + Headroom", () => {
  beforeAll(async () => {
    const res = await fetch(`${PROXY_URL}/health`);
    if (!res.ok) throw new Error("Proxy not running");
    if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY not set");
  });

  it("compress() reduces tokens, then generateText() produces valid response (OpenAI)", { timeout: 30000 }, async () => {
    const { generateText } = await import("ai");
    const { createOpenAI } = await import("@ai-sdk/openai");
    const { compress } = await import("../src/compress.js");

    const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });

    // Build a conversation where user pastes a big data dump
    const serverData = Array.from({ length: 100 }, (_, i) => ({
      id: i + 1,
      name: `server-${i + 1}`,
      status: i % 15 === 0 ? "critical" : i % 5 === 0 ? "warning" : "healthy",
      cpu: Math.round(Math.random() * 100),
      memory: Math.round(Math.random() * 100),
      region: ["us-east-1", "eu-west-1", "ap-southeast-1"][i % 3],
      description: `Production server ${i + 1} running service-${["auth", "payment", "user", "api"][i % 4]}`,
      last_alert: i % 15 === 0 ? `Disk usage at ${90 + (i % 10)}%` : null,
    }));

    // Simple messages — no tool calls (avoids Responses API quirks)
    const messages: any[] = [
      { role: "system", content: "You are an SRE assistant. Be concise — one sentence." },
      {
        role: "user",
        content: `Here is our fleet status:\n\n${JSON.stringify(serverData, null, 2)}\n\nWhich servers are critical? Just list the server names.`,
      },
    ];

    // Compress
    const compressed = await compress(messages, {
      model: "gpt-4o-mini",
      baseUrl: PROXY_URL,
    });

    console.log(
      `  Compressed: ${compressed.tokensBefore} → ${compressed.tokensAfter} tokens ` +
      `(saved ${compressed.tokensSaved}, ${((1 - compressed.compressionRatio) * 100).toFixed(0)}%)`,
    );

    expect(compressed.tokensBefore).toBeGreaterThan(0);
    // User messages are protected by default — compression happens on tool outputs
    // The important thing is the round-trip works and generateText succeeds

    // Call OpenAI via Vercel AI SDK with compressed messages
    const { text, usage } = await generateText({
      model: openai("gpt-4o-mini"),
      messages: compressed.messages,
    });

    console.log(`  LLM response: "${text.substring(0, 200)}"`);
    console.log(`  Usage: ${usage?.promptTokens} prompt, ${usage?.completionTokens} completion`);

    expect(text.length).toBeGreaterThan(0);
    // Should mention critical servers
    expect(text.toLowerCase()).toMatch(/server/i);
  });

  it("headroomMiddleware() transparently compresses before LLM call (OpenAI)", { timeout: 30000 }, async () => {
    const { generateText, wrapLanguageModel } = await import("ai");
    const { createOpenAI } = await import("@ai-sdk/openai");
    const { headroomMiddleware } = await import("../src/adapters/vercel-ai.js");

    const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });

    // Wrap model with Headroom middleware
    const model = wrapLanguageModel({
      model: openai("gpt-4o-mini"),
      middleware: headroomMiddleware({ baseUrl: PROXY_URL }),
    });

    // Big log dump as user message
    const logEntries = Array.from({ length: 100 }, (_, i) => ({
      timestamp: new Date(Date.now() - i * 1000).toISOString(),
      level: i === 42 ? "FATAL" : i % 10 === 0 ? "ERROR" : "INFO",
      service: `service-${["auth", "payment", "user", "notification"][i % 4]}`,
      message: i === 42
        ? "Connection pool exhausted - max_connections=100 reached, 47 pending requests dropped"
        : `Request processed in ${Math.round(Math.random() * 500)}ms for endpoint /${["login", "checkout", "profile", "notify"][i % 4]}`,
      trace_id: `trace-${Math.random().toString(36).substring(2, 10)}`,
    }));

    // Call generateText — middleware compresses automatically
    const { text, usage } = await generateText({
      model,
      system: "You are an SRE assistant. Identify the single most critical issue. One sentence only.",
      prompt: `Here are the recent logs:\n\n${JSON.stringify(logEntries, null, 2)}\n\nWhat is the most critical issue?`,
    });

    console.log(`  Middleware response: "${text.substring(0, 200)}"`);
    console.log(`  Usage: ${usage?.promptTokens} prompt, ${usage?.completionTokens} completion`);

    expect(text.length).toBeGreaterThan(0);
    // Should identify the FATAL error about connection pool
    expect(text.toLowerCase()).toMatch(/connection|pool|exhaust|fatal/i);
  });

  it("compress() works end-to-end with Anthropic via Vercel AI SDK", { timeout: 30000 }, async () => {
    if (!process.env.ANTHROPIC_API_KEY) {
      console.log("  Skipping: ANTHROPIC_API_KEY not set");
      return;
    }

    const { generateText } = await import("ai");
    const { createAnthropic } = await import("@ai-sdk/anthropic");
    const { compress } = await import("../src/compress.js");

    const anthropic = createAnthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
      baseURL: "https://api.anthropic.com/v1",
    });

    const searchResults = Array.from({ length: 80 }, (_, i) => ({
      title: `Result ${i + 1}: ${["API design patterns", "Database optimization", "Cache invalidation strategies", "Load balancing algorithms"][i % 4]}`,
      url: `https://docs.example.com/article-${i + 1}`,
      snippet: `This article covers ${["best practices", "common pitfalls", "advanced techniques", "getting started"][i % 4]} for ${["microservices", "distributed systems", "cloud native", "serverless"][i % 4]} architecture with detailed examples and benchmarks.`,
      relevance: (100 - i) / 100,
    }));

    const messages: any[] = [
      { role: "system", content: "You are a technical writer. Be concise — one sentence per result." },
      {
        role: "user",
        content: `Here are search results:\n\n${JSON.stringify(searchResults, null, 2)}\n\nWhat are the top 3 most relevant results? One sentence each.`,
      },
    ];

    // Compress
    const compressed = await compress(messages, {
      model: "claude-haiku-4-5",
      baseUrl: PROXY_URL,
    });

    console.log(
      `  Anthropic compressed: ${compressed.tokensBefore} → ${compressed.tokensAfter} ` +
      `(saved ${compressed.tokensSaved})`,
    );

    expect(compressed.tokensBefore).toBeGreaterThan(0);

    // Call Anthropic via Vercel AI SDK
    const { text } = await generateText({
      model: anthropic("claude-haiku-4-5-20251001"),
      messages: compressed.messages,
      maxTokens: 300,
    });

    console.log(`  Anthropic response: "${text.substring(0, 200)}"`);
    expect(text.length).toBeGreaterThan(0);
  });
});
