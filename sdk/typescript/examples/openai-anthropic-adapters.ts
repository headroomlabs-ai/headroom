/**
 * Example 11: OpenAI & Anthropic SDK Adapters
 *
 * Wrap native SDK clients directly — no Vercel AI SDK needed.
 * Messages are compressed transparently before each API call.
 *
 * Run: npx tsx examples/11-openai-anthropic-adapters.ts
 */
import { withHeadroom as withHeadroomOpenAI } from "headroom-ai/openai";
import { withHeadroom as withHeadroomAnthropic } from "headroom-ai/anthropic";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

// Same large context for both providers
const githubIssues = Array.from({ length: 40 }, (_, i) => ({
  number: 1000 + i,
  title: `Issue #${1000 + i}: ${["Fix memory leak in worker pool", "Add retry logic for API calls", "Refactor auth middleware", "Update deprecated dependencies", "Add rate limiting"][i % 5]}`,
  state: i % 8 === 0 ? "closed" : "open",
  labels: [["bug", "P1"], ["enhancement"], ["tech-debt", "P2"], ["security"], ["performance"]][i % 5],
  author: `dev${i % 7}`,
  comments: Math.floor(Math.random() * 15),
  body: `Detailed description of issue ${1000 + i}. This includes reproduction steps, expected behavior, actual behavior, environment details, and relevant log snippets. The issue was first reported on ${new Date(Date.now() - Math.random() * 90 * 86400000).toISOString().split("T")[0]}.`,
  created_at: new Date(Date.now() - Math.random() * 90 * 86400000).toISOString(),
}));

const messages = [
  {
    role: "user" as const,
    content: `Here are our open GitHub issues:\n\n${JSON.stringify(githubIssues)}\n\nPrioritize the top 5 issues we should tackle this sprint and explain why.`,
  },
];

async function main() {
  // === OpenAI with compression ===
  if (process.env.OPENAI_API_KEY) {
    console.log("=== OpenAI (compressed) ===");
    const openai = withHeadroomOpenAI(new OpenAI());

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: "You are a technical project manager. Prioritize issues for the sprint." },
        ...messages,
      ],
    });

    console.log(response.choices[0].message.content?.slice(0, 400), "...\n");
  }

  // === Anthropic with compression ===
  if (process.env.ANTHROPIC_API_KEY) {
    console.log("=== Anthropic (compressed) ===");
    const anthropic = withHeadroomAnthropic(new Anthropic());

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-5-20250929",
      max_tokens: 1024,
      messages,
    });

    const text = response.content
      .filter((b: any) => b.type === "text")
      .map((b: any) => b.text)
      .join("");
    console.log(text.slice(0, 400), "...\n");
  }

  if (!process.env.OPENAI_API_KEY && !process.env.ANTHROPIC_API_KEY) {
    console.log("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this example.");
  }
}

main().catch(console.error);
