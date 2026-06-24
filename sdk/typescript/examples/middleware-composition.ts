/**
 * Example 10: Middleware Composition
 *
 * Combine headroomMiddleware with other Vercel AI SDK middlewares.
 * Headroom compresses first, then other middlewares process the result.
 *
 * Run: npx tsx examples/10-middleware-composition.ts
 */
import { headroomMiddleware } from "headroom-ai/vercel-ai";
import { openai } from "@ai-sdk/openai";
import { generateText, wrapLanguageModel, extractReasoningMiddleware } from "ai";

// Large context that benefits from compression
const codebaseAnalysis = Array.from({ length: 40 }, (_, i) => ({
  file: `src/modules/${["auth", "billing", "users", "api", "workers"][i % 5]}/handler${Math.floor(i / 5)}.ts`,
  lines: Math.floor(Math.random() * 500 + 100),
  complexity: Math.floor(Math.random() * 30 + 1),
  dependencies: Array.from({ length: Math.floor(Math.random() * 8) }, (_, j) => `dep-${j}`),
  last_modified: new Date(Date.now() - Math.random() * 180 * 86400000).toISOString(),
  test_coverage: +(Math.random() * 100).toFixed(1),
  issues: Array.from({ length: Math.floor(Math.random() * 3) }, () =>
    ["unused import", "any type", "missing null check", "no error handling", "hardcoded value"][Math.floor(Math.random() * 5)]
  ),
}));

async function main() {
  // Stack multiple middlewares: compression + reasoning extraction
  const model = wrapLanguageModel({
    model: openai("gpt-4o"),
    middleware: [
      headroomMiddleware(), // compress large context
      extractReasoningMiddleware({ tagName: "think" }), // extract chain-of-thought
    ],
  });

  const { text, reasoning } = await generateText({
    model,
    messages: [
      {
        role: "system",
        content: `You are a code quality analyst. When analyzing, wrap your reasoning in <think> tags before giving your final answer.`,
      },
      {
        role: "user",
        content: `Analyze this codebase for quality issues:\n\n${JSON.stringify(codebaseAnalysis)}\n\nWhat are the top 3 areas that need immediate attention?`,
      },
    ],
  });

  if (reasoning) {
    console.log("=== Internal Reasoning ===");
    console.log(reasoning.slice(0, 300), "...\n");
  }

  console.log("=== Analysis ===");
  console.log(text);
}

main().catch(console.error);
