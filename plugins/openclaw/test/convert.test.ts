import { describe, expect, it } from "vitest";
import { agentToOpenAI, agentToOpenAIIndexed, normalizeAgentMessages, openAIToAgent, restoreAgentMessages, type OpenAIMessage } from "../src/convert";

describe("openAIToAgent", () => {
  it("emits toolResult content as blocks so transports can safely filter", () => {
    const messages: OpenAIMessage[] = [
      {
        role: "tool",
        content: "tool output",
        tool_call_id: "call_123",
      },
    ];

    const result = openAIToAgent(messages);
    const toolResult = result[0] as {
      role: string;
      content: Array<{ type: string; text?: string }>;
      toolCallId: string;
      tool_use_id: string;
    };

    expect(toolResult.role).toBe("toolResult");
    expect(Array.isArray(toolResult.content)).toBe(true);
    expect(toolResult.content).toEqual([{ type: "text", text: "tool output" }]);
    expect(toolResult.toolCallId).toBe("call_123");
    expect(toolResult.tool_use_id).toBe("call_123");
  });
});

describe("normalizeAgentMessages", () => {
  it("normalizes assistant string content into OpenClaw blocks", () => {
    const result = normalizeAgentMessages([
      {
        role: "assistant",
        content: "hello from headroom",
      },
    ]);

    expect(result[0]).toMatchObject({
      role: "assistant",
      content: [{ type: "text", text: "hello from headroom" }],
      api: "headroom",
      provider: "headroom",
      model: "headroom",
      stopReason: "stop",
    });
  });

  it("normalizes tool result string content into OpenClaw blocks", () => {
    const result = normalizeAgentMessages([
      {
        role: "toolResult",
        content: "tool output",
      },
    ]);

    expect(result[0]).toMatchObject({
      role: "toolResult",
      content: [{ type: "text", text: "tool output" }],
      toolCallId: "unknown",
      tool_use_id: "unknown",
      toolName: "headroom",
      isError: false,
    });
  });

  it("preserves provider thought signatures on canonical tool calls", () => {
    const result = normalizeAgentMessages([
      {
        role: "assistant",
        content: [
          {
            type: "toolCall",
            id: "call_signed",
            name: "read",
            arguments: { path: "file.ts" },
            thoughtSignature: "provider-tool-call-signature",
          },
        ],
      },
    ]);

    expect(result[0].content[0].thoughtSignature).toBe("provider-tool-call-signature");
    expect(result[0].content[0].arguments).toEqual({ path: "file.ts" });
  });

  it("normalizes legacy tool inputs without retaining a duplicate alias", () => {
    const block = {
      type: "tool_use",
      name: "read",
      input: { path: "file.ts" },
      thoughtSignature: "legacy-signature",
      providerMetadata: { opaque: true },
    };

    const result = normalizeAgentMessages([{ role: "assistant", content: [block] }]);

    expect(result[0].content[0]).toEqual({
      type: "toolCall",
      id: "unknown",
      name: "read",
      arguments: { path: "file.ts" },
      thoughtSignature: "legacy-signature",
      providerMetadata: { opaque: true },
    });
    expect(block.input).toEqual({ path: "file.ts" });
    expect(block.type).toBe("tool_use");
  });

  it("keeps existing arguments precedence while removing the legacy alias", () => {
    const result = normalizeAgentMessages([
      {
        role: "assistant",
        content: [{ type: "tool_use", name: "read", arguments: null, input: { path: "old.ts" } }],
      },
    ]);

    expect(result[0].content[0].arguments).toBeNull();
    expect(result[0].content[0]).not.toHaveProperty("input");
  });

  it("retains opaque extension fields on already canonical tool calls", () => {
    const block = {
      type: "toolCall",
      id: "call_canonical",
      name: "read",
      arguments: { path: "file.ts" },
      input: { extension: "opaque canonical metadata" },
      thoughtSignature: "canonical-signature",
      providerMetadata: { opaque: true },
    };

    const result = normalizeAgentMessages([{ role: "assistant", content: [block] }]);

    expect(result[0].content[0]).toEqual(block);
  });
});

describe("restoreAgentMessages", () => {
  it("restores compressed legacy arguments without stale input and preserves metadata", () => {
    const block = {
      type: "tool_use",
      id: "call_legacy",
      name: "read",
      input: { path: "file.ts", lines: ["long", "original", "payload"] },
      thoughtSignature: "legacy-signature",
      providerMetadata: { opaque: true },
    };
    const original = [{ role: "assistant", content: [block] }];
    const sent = agentToOpenAIIndexed(original);
    const returned = structuredClone(sent);
    returned[0].tool_calls![0].function.arguments = '{"path":"file.ts","lines":["short"]}';

    const result = restoreAgentMessages(original, sent, returned);

    expect(result[0].content[0]).toEqual({
      type: "toolCall",
      id: "call_legacy",
      name: "read",
      arguments: { path: "file.ts", lines: ["short"] },
      thoughtSignature: "legacy-signature",
      providerMetadata: { opaque: true },
    });
    expect(block.input.lines).toEqual(["long", "original", "payload"]);
    expect(agentToOpenAI(result)[0].tool_calls![0].function.arguments)
      .toBe('{"path":"file.ts","lines":["short"]}');
  });
});

describe("agentToOpenAI", () => {
  it("captures assistant metadata needed for OpenClaw round-trips", () => {
    const result = agentToOpenAI([
      {
        role: "assistant",
        content: "hello",
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-sonnet-4-5",
        stopReason: "stop",
        usage: {
          input: 1,
          output: 2,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 3,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
      },
    ]);

    expect(result[0]._headroomMeta).toMatchObject({
      api: "anthropic-messages",
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      stopReason: "stop",
    });
  });
});
