/**
 * Example 04: Tool-Calling Agent with Compression
 *
 * Build an agent that calls tools and compresses the growing context
 * automatically. Each tool result gets compressed before the next LLM call.
 *
 * Run: npx tsx examples/04-tool-calling-agent.ts
 */
import { withHeadroom } from "headroom-ai/vercel-ai";
import { openai } from "@ai-sdk/openai";
import { generateText, tool, stepCountIs } from "ai";
import { z } from "zod";

// Mock database
const mockDB: Record<string, any[]> = {
  users: Array.from({ length: 100 }, (_, i) => ({
    id: i + 1,
    name: `User ${i + 1}`,
    email: `user${i + 1}@example.com`,
    role: ["admin", "editor", "viewer"][i % 3],
    last_login: new Date(Date.now() - Math.random() * 30 * 86400000).toISOString(),
    active: i % 7 !== 0,
  })),
  logs: Array.from({ length: 200 }, (_, i) => ({
    id: i + 1,
    timestamp: new Date(Date.now() - i * 60000).toISOString(),
    level: i === 47 ? "FATAL" : i % 20 === 0 ? "ERROR" : i % 5 === 0 ? "WARN" : "INFO",
    service: ["auth", "api", "worker", "scheduler"][i % 4],
    message: `${i === 47 ? "Database connection pool exhausted" : `Processing request ${i}`}`,
    trace_id: `trace-${Math.random().toString(36).slice(2, 10)}`,
  })),
};

async function main() {
  const model = withHeadroom(openai("gpt-4o"));

  const { text, steps } = await generateText({
    model,
    stopWhen: stepCountIs(5),
    messages: [
      {
        role: "system",
        content: "You are a database admin assistant. Use the available tools to investigate issues. Report findings concisely.",
      },
      {
        role: "user",
        content: "Check the logs for any critical errors, then find which users are affected.",
      },
    ],
    tools: {
      query_logs: tool({
        description: "Query application logs. Returns matching log entries.",
        parameters: z.object({
          level: z.string().optional().describe("Filter by log level: INFO, WARN, ERROR, FATAL"),
          service: z.string().optional().describe("Filter by service name"),
          limit: z.number().optional().describe("Max results to return"),
        }),
        execute: async ({ level, service, limit }) => {
          let results = mockDB.logs;
          if (level) results = results.filter((l) => l.level === level);
          if (service) results = results.filter((l) => l.service === service);
          return results.slice(0, limit ?? 50);
        },
      }),
      query_users: tool({
        description: "Query user database. Returns matching users.",
        parameters: z.object({
          role: z.string().optional().describe("Filter by role: admin, editor, viewer"),
          active: z.boolean().optional().describe("Filter by active status"),
          limit: z.number().optional().describe("Max results"),
        }),
        execute: async ({ role, active, limit }) => {
          let results = mockDB.users;
          if (role) results = results.filter((u) => u.role === role);
          if (active !== undefined) results = results.filter((u) => u.active === active);
          return results.slice(0, limit ?? 25);
        },
      }),
    },
  });

  console.log(`Agent completed in ${steps.length} steps`);
  console.log("\nFinal answer:", text);
}

main().catch(console.error);
