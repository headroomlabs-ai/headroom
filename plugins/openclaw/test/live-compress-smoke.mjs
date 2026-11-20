#!/usr/bin/env node
/**
 * Live smoke test against a running Headroom proxy.
 * Run: node test/live-compress-smoke.mjs
 * Requires proxy at HEADROOM_PROXY_URL (default http://127.0.0.1:8787).
 */
import { agentToOpenAI, openAIToAgent } from "../dist/index.js";

const PROXY_URL = process.env.HEADROOM_PROXY_URL ?? "http://127.0.0.1:8787";

const transcript = [
  {
    role: "assistant",
    content: [
      { type: "text", text: "Analyzing screenshot." },
      {
        type: "toolCall",
        id: "call_view_1",
        name: "view_image",
        arguments: { path: "/tmp/godot.png" },
      },
    ],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-sonnet-4-5",
    stopReason: "toolUse",
  },
  {
    role: "toolResult",
    toolCallId: "call_view_1",
    toolName: "view_image",
    content: [
      {
        type: "image",
        data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
        mimeType: "image/png",
      },
    ],
  },
];

async function main() {
  const health = await fetch(`${PROXY_URL}/health`);
  if (!health.ok) {
    console.error("Proxy unhealthy:", health.status);
    process.exit(1);
  }

  const openaiMessages = agentToOpenAI(transcript);
  const viewTool = openaiMessages.find((m) => m.role === "tool" && m.name === "view_image");
  if (!viewTool?.name) {
    console.error("FAIL: plugin conversion missing tool.name before compress");
    process.exit(1);
  }

  const imageData = transcript
    .flatMap((m) => (Array.isArray(m.content) ? m.content : []))
    .filter((b) => b?.type === "image")
    .map((b) => b.data);
  const wire = JSON.stringify(openaiMessages);
  if (imageData.some((data) => wire.includes(data))) {
    console.error("FAIL: image bytes would be sent to the proxy");
    process.exit(1);
  }

  const body = {
    messages: openaiMessages,
    model: "claude-sonnet-4-5",
    config: { protect_recent: 2 },
  };

  const response = await fetch(`${PROXY_URL}/v1/compress`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-headroom-mode": "token",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const detail = await response.text();
    console.error("Compress failed:", response.status, detail.slice(0, 400));
    process.exit(1);
  }

  const data = await response.json();
  const restored = openAIToAgent(data.messages, { originals: transcript });
  const viewResult = restored.find(
    (m) => m.role === "toolResult" && m.toolName === "view_image",
  );
  const imageBlock = viewResult?.content?.find?.((b) => b?.type === "image");

  console.log("Proxy:", PROXY_URL);
  console.log("Tokens:", data.tokens_before, "→", data.tokens_after, "(saved", data.tokens_saved + ")");
  console.log("Request tool.name:", viewTool.name);
  console.log("Response has image block:", Boolean(imageBlock));

  if (!imageBlock) {
    console.error("FAIL: view_image image block lost after live compress round-trip");
    console.error("Restored toolResult content:", JSON.stringify(viewResult?.content)?.slice(0, 300));
    process.exit(1);
  }

  console.log("PASS: live compress smoke test — image payload preserved");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
