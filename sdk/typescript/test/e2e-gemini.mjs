// End-to-end: compress with headroom, send to real Gemini API, verify response.
// Run with:  GEMINI_API_KEY=... node test/e2e-gemini.mjs
//
// Build a chunky multi-turn Gemini-format conversation, compress through the
// headroom proxy, then POST the compressed contents to gemini-2.5-flash.

import { compress } from "../dist/index.js";

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("missing GEMINI_API_KEY in env");
  process.exit(2);
}

const baseUrl = process.env.HEADROOM_BASE_URL ?? "http://localhost:8788";

// Build a 100-item search-result tool output, plus a conversation around it.
const items = Array.from({ length: 100 }, (_, i) => ({
  id: `doc_${i}`,
  title: `Search result number ${i} from the documentation index`,
  snippet: `This is a moderately long snippet for result ${i} explaining what the document covers in some detail.`,
  score: 1 - i / 100,
  url: `https://example.com/docs/${i}`,
}));

// Gemini-format messages
const geminiMessages = [
  { role: "user", parts: [{ text: "You are a helpful assistant. Be concise." }] },
  {
    role: "user",
    parts: [{ text: "I just searched the docs. Here are the results — what are the top 3 by relevance?" }],
  },
  {
    role: "user",
    parts: [{ text: `Search results JSON:\n${JSON.stringify(items)}` }],
  },
];

console.log(`headroom proxy: ${baseUrl}`);
console.log(`input: ${geminiMessages.length} gemini messages, ${items.length} search items`);

// Step 1: compress through headroom
const t0 = performance.now();
const compressed = await compress(geminiMessages, {
  model: "gemini-2.5-flash",
  baseUrl,
});
const t1 = performance.now();

console.log(`\n--- compression ---`);
console.log(`tokens before: ${compressed.tokensBefore}`);
console.log(`tokens after:  ${compressed.tokensAfter}`);
console.log(`tokens saved:  ${compressed.tokensSaved}`);
console.log(`transforms:    ${compressed.transformsApplied.join(", ")}`);
console.log(`compressed shape preserved: ${
  Array.isArray(compressed.messages) &&
  compressed.messages.every((m) => "role" in m && Array.isArray(m.parts))
}`);
console.log(`compress latency: ${(t1 - t0).toFixed(1)}ms`);

// Step 2: send the compressed Gemini-shape messages to gemini-2.5-flash
const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`;
const t2 = performance.now();
const resp = await fetch(apiUrl, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-goog-api-key": apiKey,
  },
  body: JSON.stringify({ contents: compressed.messages }),
});
const t3 = performance.now();

if (!resp.ok) {
  const body = await resp.text();
  console.error(`\nGemini API error ${resp.status}: ${body.slice(0, 500)}`);
  process.exit(1);
}

const data = await resp.json();
const text = data?.candidates?.[0]?.content?.parts?.map((p) => p.text).join("") ?? "";

console.log(`\n--- gemini response ---`);
console.log(`api latency: ${(t3 - t2).toFixed(0)}ms`);
console.log(`response length: ${text.length} chars`);
console.log(`response preview:\n${text.slice(0, 600)}`);

if (!text || text.length < 10) {
  console.error("\nFAIL: response too short");
  process.exit(1);
}
console.log("\nPASS: TS SDK e2e (headroom compress + gemini-2.5-flash)");
