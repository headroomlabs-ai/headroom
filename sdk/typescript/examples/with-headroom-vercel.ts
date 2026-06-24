/**
 * Example 02: withHeadroom — Vercel AI SDK One-Liner
 *
 * Wrap any Vercel AI SDK model with automatic compression.
 * Zero config — just wrap and use.
 *
 * Run: npx tsx examples/02-with-headroom-vercel.ts
 */
import { withHeadroom } from "headroom-ai/vercel-ai";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

// Large API response from a search tool
const searchResults = Array.from({ length: 50 }, (_, i) => ({
  rank: i + 1,
  title: `Result ${i + 1}: ${["Understanding microservices", "Docker best practices", "Kubernetes scaling", "CI/CD pipelines", "Cloud architecture"][i % 5]}`,
  snippet: `This is a detailed snippet for search result ${i + 1}. It contains relevant information about the topic including technical details, code examples, and best practices that would normally consume many tokens.`,
  url: `https://example.com/article-${i + 1}`,
  score: +(Math.random() * 0.5 + 0.5).toFixed(3),
  published: "2025-03-15",
  author: `Author ${i % 10}`,
}));

async function main() {
  // One-liner: wrap the model, compression happens automatically
  const model = withHeadroom(openai("gpt-4o"));

  const { text, usage } = await generateText({
    model,
    messages: [
      { role: "system", content: "You are a research assistant. Summarize search results concisely." },
      { role: "user", content: "Search for microservices best practices" },
      {
        role: "assistant",
        content: null,
        toolInvocations: [],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_search",
            toolName: "web_search",
            result: searchResults,
          },
        ],
      },
      { role: "user", content: "What are the top 3 most relevant results and why?" },
    ],
  });

  console.log("Response:", text);
  console.log("Tokens used:", usage);
}

main().catch(console.error);
