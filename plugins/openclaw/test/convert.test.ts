import { describe, expect, it } from "vitest";
import { agentToOpenAI, normalizeAgentMessages, openAIToAgent, type OpenAIMessage } from "../src/convert";

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
