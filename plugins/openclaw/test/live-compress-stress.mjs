#!/usr/bin/env node
/**
 * Live stress test: all OpenClaw native tool shapes through real Headroom /v1/compress.
 *
 * Usage:
 *   node test/live-compress-stress.mjs
 *   HEADROOM_PROXY_URL=http://127.0.0.1:8787 node test/live-compress-stress.mjs
 *
 * No LLM calls — only POST /v1/compress (algorithmic compression).
 */
import { agentToOpenAI, openAIToAgent } from "../dist/index.js";

const PROXY_URL = process.env.HEADROOM_PROXY_URL ?? "http://127.0.0.1:8787";
const MEGA_SIZE = Number(process.env.STRESS_MEGA_SIZE ?? "80");

function largeText(n) {
  return Array.from({ length: n }, (_, i) => `line-${i}:${"z".repeat(100)}`).join("\n");
}

const SCENARIOS = [
  {
    name: "exec-large",
    messages: mkTurn("exec", "call_exec_1", [{ type: "text", text: largeText(60) }]),
  },
  {
    name: "read-medium",
    messages: mkTurn("read", "call_read_1", [{ type: "text", text: largeText(20) }]),
  },
  {
    name: "browser-text",
    messages: mkTurn("browser", "call_browser_1", [
      { type: "text", text: 'Screenshot saved to /tmp/shot.png. Use view_image next.' },
    ]),
  },
  {
    name: "view_image-png",
    messages: mkTurn("view_image", "call_view_1", [
      { type: "text", text: "Loaded image into model context." },
      {
        type: "image",
        data: Buffer.from("fake-live-stress-png").toString("base64"),
        mimeType: "image/png",
      },
    ]),
  },
  {
    name: "memory_search-json",
    messages: mkTurn("memory_search", "call_mem_1", [
      {
        type: "text",
        text: JSON.stringify({ results: [{ path: "MEMORY.md", score: 0.91 }] }),
      },
    ]),
  },
  {
    name: "web_fetch-json",
    messages: mkTurn("web_fetch", "call_fetch_1", [
      { type: "text", text: JSON.stringify({ status: 200, bytes: 4096 }) },
    ]),
  },
  {
    name: "view_image-error",
    messages: mkTurn(
      "view_image",
      "call_view_err",
      [
        {
          type: "text",
          text: JSON.stringify({
            status: "error",
            tool: "view_image",
            error: "Local media path is not under an allowed directory",
          }),
        },
      ],
      true,
    ),
  },
  {
    name: "multi-tool-assistant",
    messages: [
      {
        role: "assistant",
        content: [
          { type: "toolCall", id: "call_r1", name: "read", arguments: { path: "/tmp/a" } },
          { type: "toolCall", id: "call_b1", name: "browser", arguments: { action: "screenshot" } },
        ],
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-sonnet-4-5",
        stopReason: "toolUse",
      },
      mkToolResult("read", "call_r1", [{ type: "text", text: "aaa" }]),
      mkToolResult("browser", "call_b1", [{ type: "text", text: "saved" }]),
    ],
  },
];

function mkToolResult(toolName, callId, content, isError = false) {
  return {
    role: "toolResult",
    toolCallId: callId,
    toolName,
    content,
    isError,
    timestamp: Date.now(),
  };
}

function mkTurn(toolName, callId, content, isError = false) {
  return [
    {
      role: "assistant",
      content: [
        { type: "toolCall", id: callId, name: toolName, arguments: { stress: true } },
      ],
      api: "anthropic-messages",
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      stopReason: "toolUse",
    },
    mkToolResult(toolName, callId, content, isError),
  ];
}

function buildMega(size) {
  const out = [{ role: "user", content: "mega stress", timestamp: 1 }];
  let i = 0;
  while (out.length < size) {
    for (const s of SCENARIOS) {
      if (out.length >= size) break;
      out.push(...s.messages.map((m, j) => ({ ...m, timestamp: i * 100 + j })));
      i++;
    }
  }
  return out.slice(0, size);
}

function summarizeToolResults(messages) {
  return messages
    .filter((m) => m.role === "toolResult")
    .map((m) => {
      const blocks = Array.isArray(m.content) ? m.content : [];
      const hasImage = blocks.some((b) => b?.type === "image");
      const textLen = blocks
        .filter((b) => b?.type === "text")
        .reduce((n, b) => n + String(b.text ?? "").length, 0);
      return {
        toolCallId: m.toolCallId,
        toolName: m.toolName,
        hasImage,
        textLen,
        isError: m.isError === true,
      };
    });
}

async function compressTranscript(label, transcript) {
  const openaiIn = agentToOpenAI(transcript);
  const toolNamesIn = openaiIn.filter((m) => m.role === "tool").map((m) => m.name);
  const missingNames = toolNamesIn.filter((n) => !n);
  if (missingNames.length > 0) {
    return { label, ok: false, error: "OpenAI tool messages missing name field" };
  }

  const body = {
    messages: openaiIn,
    model: "claude-sonnet-4-5",
    config: { protect_recent: 2 },
  };

  const t0 = performance.now();
  const response = await fetch(`${PROXY_URL}/v1/compress`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-headroom-mode": "token" },
    body: JSON.stringify(body),
  });
  const elapsedMs = Math.round(performance.now() - t0);

  if (!response.ok) {
    const detail = await response.text();
    return { label, ok: false, error: `HTTP ${response.status}: ${detail.slice(0, 200)}`, elapsedMs };
  }

  const data = await response.json();
  const restored = openAIToAgent(data.messages, { originals: transcript });
  const before = summarizeToolResults(transcript);
  const after = summarizeToolResults(restored);

  const failures = [];
  for (const b of before) {
    const a = after.find((x) => x.toolCallId === b.toolCallId);
    if (!a) {
      failures.push(`${b.toolName}(${b.toolCallId}): missing after compress`);
      continue;
    }
    if (b.hasImage && !a.hasImage) {
      failures.push(`${b.toolName}(${b.toolCallId}): image block lost`);
    }
    if (b.isError && !a.isError) {
      failures.push(`${b.toolName}(${b.toolCallId}): isError flag lost`);
    }
  }

  return {
    label,
    ok: failures.length === 0,
    elapsedMs,
    tokensBefore: data.tokens_before,
    tokensAfter: data.tokens_after,
    tokensSaved: data.tokens_saved,
    toolsIn: toolNamesIn.length,
    toolsOut: after.length,
    failures,
    before,
    after,
  };
}

async function main() {
  const health = await fetch(`${PROXY_URL}/health`);
  if (!health.ok) {
    console.error("Proxy not healthy:", health.status);
    process.exit(1);
  }
  const healthJson = await health.json();
  console.log(`Proxy ${PROXY_URL} v${healthJson.version} — live compress stress\n`);

  const results = [];
  for (const scenario of SCENARIOS) {
    results.push(await compressTranscript(scenario.name, scenario.messages));
  }
  results.push(await compressTranscript(`mega-${MEGA_SIZE}`, buildMega(MEGA_SIZE)));

  let passed = 0;
  let failed = 0;
  for (const r of results) {
    const status = r.ok ? "PASS" : "FAIL";
    if (r.ok) passed++;
    else failed++;
    console.log(
      `[${status}] ${r.label} — ${r.tokensBefore ?? "?"}→${r.tokensAfter ?? "?"} tok (saved ${r.tokensSaved ?? "?"}) in ${r.elapsedMs}ms | tools ${r.toolsIn ?? "?"}→${r.toolsOut ?? "?"}`,
    );
    if (r.failures?.length) {
      for (const f of r.failures) console.log(`       ↳ ${f}`);
    }
    if (r.error) console.log(`       ↳ ${r.error}`);
  }

  console.log(`\n${passed} passed, ${failed} failed, ${results.length} total`);
  if (failed > 0) process.exit(1);
  console.log("All live compress stress scenarios passed.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
