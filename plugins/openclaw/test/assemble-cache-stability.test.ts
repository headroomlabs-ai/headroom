import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  compress: vi.fn(),
  start: vi.fn(async () => "http://127.0.0.1:8787"),
  stop: vi.fn(async () => undefined),
  logger: { debug: vi.fn(), error: vi.fn(), info: vi.fn(), warn: vi.fn() },
}));

vi.mock("headroom-ai", () => ({ compress: mocked.compress }));

vi.mock("../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = mocked.start;
    stop = mocked.stop;
  },
  defaultLogger: mocked.logger,
}));

import { normalizeAgentMessages } from "../src/convert.js";
import { HeadroomContextEngine } from "../src/engine.js";

afterEach(() => {
  mocked.compress.mockReset();
});

// An OpenClaw-shaped history: a multi-block user message, an assistant turn with a signed thinking block
// and a tool call, a large tool result, the final answer, and the next user message.
function history() {
  return [
    {
      role: "user",
      content: [
        { type: "text", text: "[Slack DM from tal]" },
        { type: "text", text: "list the pods" },
      ],
      timestamp: 1,
      senderId: "U123",
    },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "run kubectl", thinkingSignature: "sig-1" },
        { type: "text", text: "Running it." },
        { type: "toolCall", id: "t1", name: "exec", arguments: { command: "kubectl get pods -A -o json" } },
      ],
      api: "bedrock-converse-stream",
      provider: "amazon-bedrock",
      model: "claude",
      stopReason: "toolUse",
      timestamp: 2,
    },
    {
      role: "toolResult",
      toolCallId: "t1",
      toolName: "exec",
      content: [{ type: "text", text: "[" + '{"ns":"argocd","name":"a"},'.repeat(200) + "]" }],
      isError: false,
      timestamp: 3,
      details: { exitCode: 0 },
    },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "count them", thinkingSignature: "sig-2" },
        { type: "text", text: "There are 200 pods." },
      ],
      api: "bedrock-converse-stream",
      provider: "amazon-bedrock",
      model: "claude",
      stopReason: "stop",
      timestamp: 4,
    },
    { role: "user", content: [{ type: "text", text: "which namespace?" }], timestamp: 5 },
  ];
}

// A proxy that compresses only tool results, the way the real /v1/compress does for JSON tool output:
// same messages, same order, `_headroomMeta` echoed back unchanged.
function compressToolResults(messages: any[]) {
  return {
    compressed: true,
    tokensBefore: 1000,
    tokensAfter: 400,
    tokensSaved: 600,
    messages: messages.map((m) => (m.role === "tool" ? { ...m, content: "[compressed]" } : m)),
  };
}

function engine() {
  const e = new HeadroomContextEngine();
  (e as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
  return e;
}

describe("assemble() keeps the provider prompt cache", () => {
  it("returns every message it did not compress exactly as OpenClaw passed it in", async () => {
    mocked.compress.mockImplementation(async (messages: any[]) => compressToolResults(messages));
    const input = history();
    const expected = normalizeAgentMessages(history());

    const { messages } = await engine().assemble({ sessionId: "s", messages: input });

    expect(messages).toHaveLength(5);
    for (const i of [0, 1, 3, 4]) expect(messages[i]).toEqual(expected[i]);
    // The two losses the old full round trip caused, on messages nothing was compressed in:
    expect(messages[0].content).toEqual(history()[0].content); // content blocks not flattened
    expect(messages[1].content.map((b: any) => b.type)).toEqual(["thinking", "text", "toolCall"]); // thinking kept
    expect(messages[3].content.map((b: any) => b.type)).toEqual(["thinking", "text"]);
  });

  it("keeps the history before the first compressed message byte-identical", async () => {
    mocked.compress.mockImplementation(async (messages: any[]) => compressToolResults(messages));

    const { messages } = await engine().assemble({ sessionId: "s", messages: history() });

    expect(JSON.stringify(messages.slice(0, 2))).toBe(JSON.stringify(normalizeAgentMessages(history()).slice(0, 2)));
  });

  it("replaces only the text of a compressed tool result and keeps its other fields and blocks", async () => {
    mocked.compress.mockImplementation(async (messages: any[]) => compressToolResults(messages));
    const input = history();
    input[2].content.push({ type: "image", data: "aGk=", mimeType: "image/png" } as any);

    const { messages } = await engine().assemble({ sessionId: "s", messages: input });

    expect(messages[2]).toEqual({
      ...normalizeAgentMessages([input[2]])[0],
      content: [
        { type: "text", text: "[compressed]" },
        { type: "image", data: "aGk=", mimeType: "image/png" },
      ],
    });
    expect(messages[2].details).toEqual({ exitCode: 0 });
  });

  it("follows a result the proxy shortened to fit the token budget", async () => {
    mocked.compress.mockImplementation(async (messages: any[]) => {
      const r = compressToolResults(messages);
      return { ...r, messages: r.messages.slice(2) }; // rolling window dropped the two oldest
    });
    const expected = normalizeAgentMessages(history());

    const { messages } = await engine().assemble({ sessionId: "s", messages: history() });

    expect(messages).toHaveLength(3);
    expect(messages[0].toolCallId).toBe("t1");
    expect(messages[0].content).toEqual([{ type: "text", text: "[compressed]" }]);
    expect(messages[1]).toEqual(expected[3]);
    expect(messages[2]).toEqual(expected[4]);
  });

  it("converts a message the proxy added itself the same way as before", async () => {
    mocked.compress.mockImplementation(async (messages: any[]) => {
      const r = compressToolResults(messages);
      return { ...r, messages: [{ role: "user", content: "[3 earlier messages omitted]" }, ...r.messages] };
    });

    const { messages } = await engine().assemble({ sessionId: "s", messages: history() });

    expect(messages).toHaveLength(6);
    expect(messages[0]).toMatchObject({ role: "user", content: "[3 earlier messages omitted]" });
  });

  it("applies compressed tool-call arguments without dropping the assistant's thinking", async () => {
    mocked.compress.mockImplementation(async (messages: any[]) => ({
      ...compressToolResults(messages),
      messages: messages.map((m) =>
        m.role === "assistant" && m.tool_calls
          ? { ...m, tool_calls: [{ ...m.tool_calls[0], function: { ...m.tool_calls[0].function, arguments: '{"command":"kubectl get pods"}' } }] }
          : m,
      ),
    }));

    const { messages } = await engine().assemble({ sessionId: "s", messages: history() });

    expect(messages[1].content.map((b: any) => b.type)).toEqual(["thinking", "text", "toolCall"]);
    expect(messages[1].content[2].arguments).toEqual({ command: "kubectl get pods" });
  });

  it("never leaks its bookkeeping into the messages it returns", async () => {
    mocked.compress.mockImplementation(async (messages: any[]) => compressToolResults(messages));

    const { messages } = await engine().assemble({ sessionId: "s", messages: history() });

    expect(JSON.stringify(messages)).not.toContain("headroomSourceIndex");
  });
});
