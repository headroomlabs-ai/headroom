/**
 * Example 05: Multi-Provider Compression
 *
 * Use the same withHeadroom wrapper across different providers.
 * Headroom compresses identically regardless of which LLM you use.
 *
 * Run: npx tsx examples/05-multi-provider.ts
 */
import { withHeadroom } from "headroom-ai/vercel-ai";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

// Large JSON dataset — same data, different models
const salesData = Array.from({ length: 60 }, (_, i) => ({
  quarter: `Q${(i % 4) + 1} ${2023 + Math.floor(i / 4)}`,
  region: ["North America", "Europe", "Asia Pacific", "Latin America"][i % 4],
  product: ["Enterprise", "Pro", "Starter"][i % 3],
  revenue: Math.round(Math.random() * 1000000),
  customers: Math.floor(Math.random() * 500),
  churn_rate: +(Math.random() * 0.15).toFixed(3),
  nps_score: Math.floor(Math.random() * 40 + 60),
  support_tickets: Math.floor(Math.random() * 200),
  avg_response_time_hrs: +(Math.random() * 24).toFixed(1),
}));

const messages = [
  {
    role: "system" as const,
    content: "You are a business analyst. Analyze the data and provide key insights. Be concise — bullet points preferred.",
  },
  {
    role: "user" as const,
    content: `Here is our sales data:\n${JSON.stringify(salesData)}\n\nWhat are the 3 most important trends?`,
  },
];

async function main() {
  // GPT-4o with compression
  console.log("=== OpenAI GPT-4o ===");
  const gpt = withHeadroom(openai("gpt-4o"));
  const gptResult = await generateText({ model: gpt, messages });
  console.log(gptResult.text);
  console.log(`Tokens: ${gptResult.usage.promptTokens} prompt, ${gptResult.usage.completionTokens} completion\n`);

  // GPT-4o-mini with compression (cheaper, faster)
  console.log("=== OpenAI GPT-4o-mini ===");
  const mini = withHeadroom(openai("gpt-4o-mini"));
  const miniResult = await generateText({ model: mini, messages });
  console.log(miniResult.text);
  console.log(`Tokens: ${miniResult.usage.promptTokens} prompt, ${miniResult.usage.completionTokens} completion\n`);
}

main().catch(console.error);
