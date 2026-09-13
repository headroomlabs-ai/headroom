/**
 * Realistic OpenClaw native tool transcript fixtures for stress testing.
 * Shapes mirror production SQLite transcripts (text JSON-in-text, image blocks, errors).
 */

export type ToolPayloadKind = "text" | "json" | "image" | "large" | "error" | "mixed";

export interface NativeToolScenario {
  toolName: string;
  payloadKind: ToolPayloadKind;
  /** Whether image/toolCall blocks must survive lossy compress round-trip. */
  mustPreserveStructured: boolean;
  buildTurn: (index: number) => unknown[];
}

const ASSISTANT_META = {
  api: "anthropic-messages",
  provider: "anthropic",
  model: "claude-sonnet-4-5",
  stopReason: "toolUse" as const,
};

function assistantToolCall(toolName: string, callId: string, args: Record<string, unknown>, ts: number) {
  return {
    role: "assistant",
    content: [
      { type: "thinking", thinking: `Planning ${toolName} call` },
      { type: "text", text: `Running ${toolName}.` },
      { type: "toolCall", id: callId, name: toolName, arguments: args },
    ],
    ...ASSISTANT_META,
    timestamp: ts,
  };
}

function toolResult(
  toolName: string,
  callId: string,
  content: unknown[],
  ts: number,
  isError = false,
) {
  return {
    role: "toolResult",
    toolCallId: callId,
    toolName,
    content,
    isError,
    timestamp: ts,
  };
}

function jsonResult(toolName: string, callId: string, payload: unknown, ts: number) {
  return toolResult(toolName, callId, [{ type: "text", text: JSON.stringify(payload, null, 2) }], ts);
}

function largeText(lines: number): string {
  return Array.from({ length: lines }, (_, i) => `line-${i}:${"x".repeat(120)}`).join("\n");
}

export const NATIVE_TOOL_SCENARIOS: NativeToolScenario[] = [
  {
    toolName: "exec",
    payloadKind: "large",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_exec_${i}`;
      return [
        assistantToolCall("exec", id, { command: "ls -la /srv/openclaw" }, i * 10 + 1),
        toolResult("exec", id, [{ type: "text", text: largeText(80) }], i * 10 + 2),
      ];
    },
  },
  {
    toolName: "read",
    payloadKind: "text",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_read_${i}`;
      return [
        assistantToolCall("read", id, { path: "~/.openclaw/openclaw.json" }, i * 10 + 1),
        toolResult("read", id, [{ type: "text", text: largeText(40) }], i * 10 + 2),
      ];
    },
  },
  {
    toolName: "write",
    payloadKind: "text",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_write_${i}`;
      return [
        assistantToolCall("write", id, { path: "/tmp/out.txt", content: "hello" }, i * 10 + 1),
        toolResult("write", id, [{ type: "text", text: "Wrote 5 bytes to /tmp/out.txt" }], i * 10 + 2),
      ];
    },
  },
  {
    toolName: "edit",
    payloadKind: "text",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_edit_${i}`;
      return [
        assistantToolCall("edit", id, { path: "/tmp/out.txt", oldText: "a", newText: "b" }, i * 10 + 1),
        toolResult("edit", id, [{ type: "text", text: "Applied edit to /tmp/out.txt" }], i * 10 + 2),
      ];
    },
  },
  {
    toolName: "apply_patch",
    payloadKind: "text",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_patch_${i}`;
      return [
        assistantToolCall("apply_patch", id, { patch: "*** Begin Patch\n*** End Patch" }, i * 10 + 1),
        toolResult("apply_patch", id, [{ type: "text", text: "Patch applied successfully." }], i * 10 + 2),
      ];
    },
  },
  {
    toolName: "browser",
    payloadKind: "text",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_browser_${i}`;
      return [
        assistantToolCall(
          "browser",
          id,
          { action: "screenshot", path: `/tmp/shot-${i}.png` },
          i * 10 + 1,
        ),
        toolResult(
          "browser",
          id,
          [
            {
              type: "text",
              text: `Screenshot saved to /srv/openclaw/media/outbound/shot-${i}.png`,
            },
          ],
          i * 10 + 2,
        ),
      ];
    },
  },
  {
    toolName: "view_image",
    payloadKind: "image",
    mustPreserveStructured: true,
    buildTurn: (i) => {
      const id = `call_view_${i}`;
      const data = Buffer.from(`fake-png-bytes-${i}`).toString("base64");
      return [
        assistantToolCall("view_image", id, { path: `/tmp/shot-${i}.png` }, i * 10 + 1),
        toolResult(
          "view_image",
          id,
          [
            {
              type: "text",
              text: "Loaded 1 image into private model context for inspection.",
            },
            { type: "image", data, mimeType: "image/png" },
          ],
          i * 10 + 2,
        ),
      ];
    },
  },
  {
    toolName: "web_search",
    payloadKind: "json",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_web_search_${i}`;
      return [
        assistantToolCall("web_search", id, { query: "openclaw headroom compression" }, i * 10 + 1),
        jsonResult(
          "web_search",
          id,
          { results: [{ title: "Headroom docs", url: "https://docs.headroomlabs.ai", score: 0.9 }] },
          i * 10 + 2,
        ),
      ];
    },
  },
  {
    toolName: "web_fetch",
    payloadKind: "json",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_web_fetch_${i}`;
      return [
        assistantToolCall("web_fetch", id, { url: "https://example.com" }, i * 10 + 1),
        jsonResult("web_fetch", id, { status: 200, content: "<html>...</html>" }, i * 10 + 2),
      ];
    },
  },
  {
    toolName: "memory_search",
    payloadKind: "json",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_memory_${i}`;
      return [
        assistantToolCall("memory_search", id, { query: "godot dashboard" }, i * 10 + 1),
        jsonResult(
          "memory_search",
          id,
          { results: [{ path: "MEMORY.md", score: 0.82, snippet: "Godot session notes..." }] },
          i * 10 + 2,
        ),
      ];
    },
  },
  {
    toolName: "sessions_list",
    payloadKind: "json",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_sessions_list_${i}`;
      return [
        assistantToolCall("sessions_list", id, {}, i * 10 + 1),
        jsonResult("sessions_list", id, { sessions: [{ id: "agent:main:main", status: "active" }] }, i * 10 + 2),
      ];
    },
  },
  {
    toolName: "sessions_history",
    payloadKind: "json",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_sessions_hist_${i}`;
      return [
        assistantToolCall("sessions_history", id, { sessionKey: "agent:main:main" }, i * 10 + 1),
        jsonResult("sessions_history", id, { messages: [{ role: "user", content: "hi" }] }, i * 10 + 2),
      ];
    },
  },
  {
    toolName: "process",
    payloadKind: "text",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_process_${i}`;
      return [
        assistantToolCall("process", id, { action: "list" }, i * 10 + 1),
        toolResult("process", id, [{ type: "text", text: "No background processes." }], i * 10 + 2),
      ];
    },
  },
  {
    toolName: "headroom_retrieve",
    payloadKind: "text",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_hr_ret_${i}`;
      return [
        assistantToolCall("headroom_retrieve", id, { hash: "abc123" }, i * 10 + 1),
        toolResult("headroom_retrieve", id, [{ type: "text", text: "Original tool output restored." }], i * 10 + 2),
      ];
    },
  },
  {
    toolName: "view_image",
    payloadKind: "error",
    mustPreserveStructured: false,
    buildTurn: (i) => {
      const id = `call_view_err_${i}`;
      return [
        assistantToolCall("view_image", id, { path: "/tmp/not-allowed.png" }, i * 10 + 1),
        toolResult(
          "view_image",
          id,
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
          i * 10 + 2,
          true,
        ),
      ];
    },
  },
  {
    toolName: "assistant",
    payloadKind: "mixed",
    mustPreserveStructured: true,
    buildTurn: (i) => {
      const idA = `call_multi_a_${i}`;
      const idB = `call_multi_b_${i}`;
      return [
        {
          role: "assistant",
          content: [
            { type: "text", text: "I'll read and screenshot." },
            { type: "toolCall", id: idA, name: "read", arguments: { path: "/tmp/a.txt" } },
            { type: "toolCall", id: idB, name: "browser", arguments: { action: "screenshot" } },
          ],
          ...ASSISTANT_META,
          timestamp: i * 10 + 1,
        },
        toolResult("read", idA, [{ type: "text", text: "file contents" }], i * 10 + 2),
        toolResult("browser", idB, [{ type: "text", text: "saved screenshot" }], i * 10 + 3),
      ];
    },
  },
];

export function buildToolScenarioTranscript(scenario: NativeToolScenario, index = 0): unknown[] {
  return [
    { role: "user", content: `Stress test ${scenario.toolName} #${index}`, timestamp: index * 10 },
    ...scenario.buildTurn(index),
  ];
}

/** Mega transcript: every native tool once, then repeated to reach target message count. */
export function buildMegaTranscript(targetMessages = 120): unknown[] {
  const messages: unknown[] = [
    { role: "user", content: "Begin mega native-tool stress transcript.", timestamp: 0 },
  ];
  let cycle = 0;
  while (messages.length < targetMessages) {
    for (const scenario of NATIVE_TOOL_SCENARIOS) {
      if (messages.length >= targetMessages) break;
      messages.push(...scenario.buildTurn(cycle));
    }
    cycle++;
  }
  return messages.slice(0, targetMessages);
}

export function extractToolResults(messages: unknown[]): Array<{
  toolName: string;
  content: unknown;
  isError?: boolean;
}> {
  return messages
    .filter((m): m is Record<string, unknown> => typeof m === "object" && m !== null)
    .filter((m) => m.role === "toolResult")
    .map((m) => ({
      toolName: String(m.toolName ?? ""),
      content: m.content,
      isError: m.isError === true,
    }));
}

export function extractOpenAiToolNames(openaiMessages: Array<{ role?: string; name?: string }>): string[] {
  return openaiMessages
    .filter((m) => m.role === "tool" && typeof m.name === "string")
    .map((m) => m.name as string);
}
