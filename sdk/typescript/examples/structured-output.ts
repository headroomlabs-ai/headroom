/**
 * Example 09: Structured Output with Compression
 *
 * Compress large context, then extract structured data with generateObject.
 * Compression preserves the signal needed for accurate extraction.
 *
 * Run: npx tsx examples/09-structured-output.ts
 */
import { withHeadroom } from "headroom-ai/vercel-ai";
import { openai } from "@ai-sdk/openai";
import { generateText, Output } from "ai";
import { z } from "zod";

// Simulate a large incident report with lots of noise
const incidentLogs = Array.from({ length: 150 }, (_, i) => {
  if (i === 73) {
    return {
      timestamp: "2025-06-15T14:32:17Z",
      level: "FATAL",
      service: "payment-service",
      message: "Circuit breaker OPEN: downstream timeout after 30s. Affected: POST /api/v2/charges. Error: ETIMEDOUT 10.0.3.42:5432",
      trace_id: "trace-incident-001",
    };
  }
  if (i === 89) {
    return {
      timestamp: "2025-06-15T14:33:01Z",
      level: "ERROR",
      service: "payment-service",
      message: "Retry exhausted for transaction txn_abc123. Customer: cust_xyz. Amount: $249.99. Gateway: stripe. Error: upstream_timeout",
      trace_id: "trace-incident-001",
    };
  }
  return {
    timestamp: new Date(Date.now() - (150 - i) * 10000).toISOString(),
    level: "INFO",
    service: ["payment-service", "auth-service", "api-gateway", "notification-service"][i % 4],
    message: `Routine: processed request ${i}, latency=${Math.floor(Math.random() * 100)}ms`,
    trace_id: `trace-${Math.random().toString(36).slice(2, 8)}`,
  };
});

// Schema for structured incident extraction
const IncidentReport = z.object({
  severity: z.enum(["critical", "high", "medium", "low"]),
  title: z.string().describe("Short incident title"),
  rootCause: z.string().describe("Root cause analysis"),
  affectedServices: z.array(z.string()),
  affectedEndpoints: z.array(z.string()),
  timeline: z.array(
    z.object({
      time: z.string(),
      event: z.string(),
    }),
  ),
  impact: z.string().describe("Customer/business impact"),
  suggestedFix: z.string(),
});

async function main() {
  const model = withHeadroom(openai("gpt-4o"));

  console.log(`Input: ${incidentLogs.length} log entries (${JSON.stringify(incidentLogs).length} chars)`);

  const { output: report } = await generateText({
    model,
    output: Output.object({ schema: IncidentReport }),
    messages: [
      {
        role: "system",
        content: "You are an SRE incident commander. Analyze logs and produce structured incident reports.",
      },
      {
        role: "user",
        content: `Analyze these production logs and create an incident report:\n\n${JSON.stringify(incidentLogs)}`,
      },
    ],
  });

  if (!report) {
    console.error("No structured output generated.");
    return;
  }

  console.log("\n=== Incident Report ===");
  console.log(`Severity: ${report.severity}`);
  console.log(`Title: ${report.title}`);
  console.log(`Root cause: ${report.rootCause}`);
  console.log(`Affected services: ${report.affectedServices.join(", ")}`);
  console.log(`Affected endpoints: ${report.affectedEndpoints.join(", ")}`);
  console.log(`Impact: ${report.impact}`);
  console.log(`\nTimeline:`);
  for (const event of report.timeline) {
    console.log(`  ${event.time} — ${event.event}`);
  }
  console.log(`\nSuggested fix: ${report.suggestedFix}`);
}

main().catch(console.error);
