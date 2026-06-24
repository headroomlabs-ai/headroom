/**
 * Example 06: Simulation (Dry Run)
 *
 * See exactly what compression would do without calling the LLM.
 * Useful for debugging, cost estimation, and understanding compression behavior.
 *
 * Run: npx tsx examples/06-simulation-dry-run.ts
 */
import { simulate } from "headroom-ai";

// Different types of content to analyze
const logEntries = Array.from({ length: 100 }, (_, i) => ({
  timestamp: new Date(Date.now() - i * 30000).toISOString(),
  level: i === 67 ? "FATAL" : i % 15 === 0 ? "ERROR" : "INFO",
  service: "payment-gateway",
  message: i === 67
    ? "Connection pool exhausted: max_connections=100, active=100, waiting=47"
    : `Processing transaction txn_${Math.random().toString(36).slice(2, 10)}`,
  trace_id: `trace-${i}`,
  duration_ms: Math.floor(Math.random() * 500),
}));

const messages = [
  {
    role: "system" as const,
    content: "You are an SRE assistant analyzing production logs.",
  },
  {
    role: "user" as const,
    content: "Check the payment gateway logs for issues",
  },
  {
    role: "assistant" as const,
    content: null,
    tool_calls: [
      { id: "call_1", type: "function" as const, function: { name: "get_logs", arguments: '{"service":"payment-gateway","limit":100}' } },
    ],
  },
  {
    role: "tool" as const,
    content: JSON.stringify(logEntries),
    tool_call_id: "call_1",
  },
  {
    role: "user" as const,
    content: "What's wrong? Is there a critical issue?",
  },
];

async function main() {
  console.log("Running compression simulation (no LLM call)...\n");

  const sim = await simulate(messages, { model: "gpt-4o" });

  console.log("=== Simulation Results ===");
  console.log(`Tokens before:  ${sim.tokensBefore}`);
  console.log(`Tokens after:   ${sim.tokensAfter}`);
  console.log(`Tokens saved:   ${sim.tokensSaved}`);
  console.log(`Estimated savings: ${sim.estimatedSavings}`);
  console.log(`\nTransforms applied: ${sim.transforms?.join(", ") || "none"}`);

  if (sim.wasteSignals && Object.keys(sim.wasteSignals).length > 0) {
    console.log("\nWaste signals detected:");
    for (const [signal, tokens] of Object.entries(sim.wasteSignals)) {
      if (tokens > 0) console.log(`  ${signal}: ${tokens} tokens`);
    }
  }

  if (sim.blockBreakdown && Object.keys(sim.blockBreakdown).length > 0) {
    console.log("\nBlock breakdown:");
    for (const [kind, count] of Object.entries(sim.blockBreakdown)) {
      console.log(`  ${kind}: ${count}`);
    }
  }

  console.log(`\nCache alignment score: ${sim.cacheAlignmentScore}`);
  console.log(`Stable prefix hash: ${sim.stablePrefixHash || "none"}`);
}

main().catch(console.error);
