/**
 * Example 12: CCR Retrieve — Lossless Compression
 *
 * Headroom compresses aggressively but stores originals.
 * When the LLM needs full details, it calls headroom_retrieve.
 * Nothing is ever thrown away.
 *
 * Run: npx tsx examples/12-ccr-retrieve.ts
 */
import { HeadroomClient } from "headroom-ai";

// Large dataset that will trigger CCR (Compress-Cache-Retrieve)
const auditLog = Array.from({ length: 200 }, (_, i) => ({
  id: i + 1,
  timestamp: new Date(Date.now() - i * 60000).toISOString(),
  actor: `user_${Math.floor(Math.random() * 20)}`,
  action: ["login", "update_profile", "delete_record", "export_data", "change_role", "api_call"][i % 6],
  resource: `resource_${Math.floor(Math.random() * 50)}`,
  ip_address: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
  user_agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
  success: i % 23 !== 0,
  details: i % 23 === 0
    ? `FAILED: Unauthorized access attempt to admin resource. IP flagged.`
    : `Routine ${["login", "update_profile", "delete_record", "export_data", "change_role", "api_call"][i % 6]} operation completed.`,
}));

async function main() {
  const client = new HeadroomClient();

  // Compress the large audit log
  const result = await client.compress(
    [
      { role: "system", content: "You are a security analyst." },
      { role: "user", content: "Review the audit log" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "get_audit_log", arguments: '{"limit":200}' } },
        ],
      },
      { role: "tool", content: JSON.stringify(auditLog), tool_call_id: "call_1" },
      { role: "user", content: "Find any suspicious activity" },
    ] as any[],
    { model: "gpt-4o" },
  );

  console.log(`Compressed: ${result.tokensBefore} → ${result.tokensAfter} tokens`);
  console.log(`Saved: ${result.tokensSaved} tokens (${((1 - result.compressionRatio) * 100).toFixed(0)}%)`);
  console.log(`CCR hashes: ${result.ccrHashes.length}`);

  // The LLM can retrieve full originals via CCR
  if (result.ccrHashes.length > 0) {
    console.log("\n=== Retrieving originals via CCR ===");
    for (const hash of result.ccrHashes) {
      try {
        const original = await client.retrieve(hash);
        console.log(`Hash ${hash}:`);
        console.log(`  Original tokens: ${(original as any).originalTokens}`);
        console.log(`  Tool: ${(original as any).toolName}`);
        console.log(`  Retrievals: ${(original as any).retrievalCount}`);
      } catch (e: any) {
        console.log(`  Could not retrieve ${hash}: ${e.message}`);
      }
    }

    // Search within compressed content
    console.log("\n=== Searching within compressed content ===");
    try {
      const search = await client.retrieve(result.ccrHashes[0], {
        query: "unauthorized access",
      });
      console.log("Search results:", JSON.stringify(search, null, 2).slice(0, 300));
    } catch (e: any) {
      console.log(`Search error: ${e.message}`);
    }
  }
}

main().catch(console.error);
