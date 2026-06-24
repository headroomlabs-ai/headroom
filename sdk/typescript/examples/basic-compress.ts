/**
 * Example 01: Basic Compression
 *
 * Compress messages before sending to any LLM.
 * Works with any message format — OpenAI, Anthropic, Vercel AI SDK, Gemini.
 *
 * Run: npx tsx examples/01-basic-compress.ts
 */
import { compress } from "headroom-ai";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

// Simulate a large tool output — 80 server status records
const serverFleet = Array.from({ length: 80 }, (_, i) => ({
  id: i + 1,
  hostname: `web-${String(i + 1).padStart(3, "0")}.prod.internal`,
  status: i === 42 ? "critical" : i % 15 === 0 ? "warning" : "healthy",
  cpu_percent: Math.round(Math.random() * 100),
  memory_mb: Math.round(Math.random() * 16384),
  region: ["us-east-1", "eu-west-1", "ap-southeast-1"][i % 3],
  last_heartbeat: "2025-06-15T10:30:00Z",
  uptime_hours: Math.floor(Math.random() * 8760),
  active_connections: Math.floor(Math.random() * 1000),
}));

const messages = [
  {
    role: "system" as const,
    content: "You are a DevOps assistant. Analyze infrastructure data and report issues concisely.",
  },
  { role: "user" as const, content: "Show me all servers" },
  {
    role: "assistant" as const,
    content: null,
    tool_calls: [
      { id: "call_1", type: "function" as const, function: { name: "list_servers", arguments: "{}" } },
    ],
  },
  {
    role: "tool" as const,
    content: JSON.stringify(serverFleet),
    tool_call_id: "call_1",
  },
  { role: "user" as const, content: "Which servers need attention? Be specific." },
];

async function main() {
  // Compress the conversation
  const result = await compress(messages, { model: "gpt-4o" });

  console.log(`Tokens: ${result.tokensBefore} → ${result.tokensAfter}`);
  console.log(`Saved: ${result.tokensSaved} tokens (${((1 - result.compressionRatio) * 100).toFixed(0)}%)`);
  console.log(`Transforms: ${result.transformsApplied.join(", ")}`);

  // Use compressed messages with Vercel AI SDK
  const { text } = await generateText({
    model: openai("gpt-4o"),
    messages: result.messages,
  });

  console.log("\nAssistant:", text);
}

main().catch(console.error);
