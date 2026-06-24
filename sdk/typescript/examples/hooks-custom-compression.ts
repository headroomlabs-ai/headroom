/**
 * Example 07: Custom Compression Hooks
 *
 * Use CompressionHooks to customize what gets compressed and observe results.
 * Hooks run client-side before/after the proxy compression.
 *
 * Run: npx tsx examples/07-hooks-custom-compression.ts
 */
import { compress, CompressionHooks } from "headroom-ai";
import type { CompressContext, CompressEvent } from "headroom-ai";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

// Track compression stats across multiple calls
const stats = {
  calls: 0,
  totalSaved: 0,
  transforms: new Map<string, number>(),
};

class ObservabilityHooks extends CompressionHooks {
  /**
   * Add context before compression — inject hints for the compressor.
   */
  preCompress(messages: any[], ctx: CompressContext) {
    console.log(`[hook] Pre-compress: ${messages.length} messages, model=${ctx.model}`);
    console.log(`[hook] User query: "${ctx.userQuery.slice(0, 60)}..."`);
    console.log(`[hook] Tool calls in context: ${ctx.toolCalls.join(", ") || "none"}`);
    return messages;
  }

  /**
   * Set per-message compression biases.
   * Higher bias = preserve more. Lower bias = compress more aggressively.
   */
  computeBiases(messages: any[], _ctx: CompressContext) {
    const biases: Record<number, number> = {};
    for (let i = 0; i < messages.length; i++) {
      // Always preserve system messages fully
      if (messages[i].role === "system") {
        biases[i] = 2.0;
      }
      // Preserve the last user message (the actual question)
      if (i === messages.length - 1 && messages[i].role === "user") {
        biases[i] = 1.5;
      }
    }
    return biases;
  }

  /**
   * Observe compression results — log and track stats.
   */
  postCompress(event: CompressEvent) {
    stats.calls++;
    stats.totalSaved += event.tokensSaved;
    for (const t of event.transformsApplied) {
      stats.transforms.set(t, (stats.transforms.get(t) ?? 0) + 1);
    }

    console.log(`[hook] Post-compress: ${event.tokensBefore} → ${event.tokensAfter} (saved ${event.tokensSaved})`);
    console.log(`[hook] Ratio: ${(event.compressionRatio * 100).toFixed(1)}%`);
    console.log(`[hook] Transforms: ${event.transformsApplied.join(", ")}`);
    if (event.ccrHashes.length > 0) {
      console.log(`[hook] CCR hashes (retrievable): ${event.ccrHashes.join(", ")}`);
    }
  }
}

// Large structured data
const inventory = Array.from({ length: 75 }, (_, i) => ({
  sku: `SKU-${String(i + 1).padStart(4, "0")}`,
  name: `Product ${i + 1}`,
  category: ["Electronics", "Clothing", "Home", "Sports", "Books"][i % 5],
  price: +(Math.random() * 200 + 10).toFixed(2),
  stock: Math.floor(Math.random() * 500),
  warehouse: ["NYC", "LAX", "ORD", "DFW"][i % 4],
  reorder_point: Math.floor(Math.random() * 50 + 10),
  last_sold: new Date(Date.now() - Math.random() * 7 * 86400000).toISOString(),
}));

async function main() {
  const hooks = new ObservabilityHooks();

  console.log("=== Call 1: Inventory analysis ===\n");
  const result1 = await compress(
    [
      { role: "system", content: "You are an inventory management assistant." },
      { role: "user", content: "Show me inventory" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "c1", type: "function", function: { name: "get_inventory", arguments: "{}" } }],
      },
      { role: "tool", content: JSON.stringify(inventory), tool_call_id: "c1" },
      { role: "user", content: "Which items are below reorder point and need restocking?" },
    ],
    { model: "gpt-4o", hooks },
  );

  // Use compressed messages
  const { text } = await generateText({
    model: openai("gpt-4o"),
    messages: result1.messages,
  });
  console.log("\nAnswer:", text.slice(0, 200), "...\n");

  console.log("=== Cumulative Stats ===");
  console.log(`Total calls: ${stats.calls}`);
  console.log(`Total tokens saved: ${stats.totalSaved}`);
  console.log("Transform frequency:");
  for (const [name, count] of stats.transforms) {
    console.log(`  ${name}: ${count}x`);
  }
}

main().catch(console.error);
