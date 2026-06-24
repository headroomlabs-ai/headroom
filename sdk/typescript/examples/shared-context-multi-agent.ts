/**
 * Example 08: SharedContext — Multi-Agent Compressed Handoff
 *
 * Agent A researches a topic and stores compressed results.
 * Agent B reads the compressed context and acts on it.
 * Saves 70-90% of tokens on inter-agent communication.
 *
 * Run: npx tsx examples/08-shared-context-multi-agent.ts
 */
import { SharedContext } from "headroom-ai";
import { withHeadroom } from "headroom-ai/vercel-ai";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

// Simulate a research agent's output — large structured data
const researchOutput = {
  topic: "Kubernetes autoscaling strategies",
  sources: Array.from({ length: 30 }, (_, i) => ({
    title: `Source ${i + 1}: ${["HPA deep dive", "VPA configuration", "Cluster autoscaler", "KEDA event-driven", "Custom metrics"][i % 5]}`,
    content: `Detailed technical content about ${["horizontal pod autoscaling", "vertical pod autoscaling", "node autoscaling", "event-driven scaling", "custom metrics scaling"][i % 5]}. This section covers implementation details, best practices, common pitfalls, and real-world examples from production deployments at scale.`,
    relevance: +(Math.random() * 0.5 + 0.5).toFixed(2),
    citations: Math.floor(Math.random() * 100),
  })),
  summary: "Kubernetes offers multiple autoscaling strategies at different levels: HPA for pod replicas, VPA for resource requests, Cluster Autoscaler for nodes, and KEDA for event-driven workloads.",
};

async function main() {
  const ctx = new SharedContext({ model: "gpt-4o", maxEntries: 50 });
  const model = withHeadroom(openai("gpt-4o"));

  // === Agent A: Research Agent ===
  console.log("=== Agent A: Research ===");

  const entry = await ctx.put("k8s-scaling-research", JSON.stringify(researchOutput), {
    agent: "researcher",
  });

  console.log(`Stored: ${entry.originalTokens} tokens → ${entry.compressedTokens} tokens`);
  console.log(`Savings: ${entry.savingsPercent.toFixed(0)}%`);
  console.log(`Transforms: ${entry.transforms.join(", ")}`);

  // === Agent B: Writer Agent ===
  console.log("\n=== Agent B: Writer ===");

  // Read compressed context (80% smaller)
  const compressed = ctx.get("k8s-scaling-research");
  console.log(`Reading compressed context (${compressed?.length ?? 0} chars)`);

  const { text } = await generateText({
    model,
    messages: [
      {
        role: "system",
        content: "You are a technical writer. Create a concise blog post outline from research data.",
      },
      {
        role: "user",
        content: `Based on this research, create a blog post outline:\n\n${compressed}`,
      },
    ],
  });

  console.log("\nBlog outline:", text);

  // === Stats ===
  console.log("\n=== SharedContext Stats ===");
  const stats = ctx.stats();
  console.log(`Entries: ${stats.entries}`);
  console.log(`Original tokens: ${stats.totalOriginalTokens}`);
  console.log(`Compressed tokens: ${stats.totalCompressedTokens}`);
  console.log(`Total saved: ${stats.totalTokensSaved} (${stats.savingsPercent.toFixed(0)}%)`);
}

main().catch(console.error);
