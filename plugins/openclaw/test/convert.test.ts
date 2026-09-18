import { describe, expect, it } from "vitest";
import {
  agentToOpenAI,
  estimateRoughTokens,
  normalizeAgentMessages,
  openAIToAgent,
  type OpenAIMessage,
} from "../src/convert";
import { IMAGE_TOKEN_ESTIMATE } from "../src/content-blocks";

const IMAGE_BLOCK = { type: "image", data: "aGVsbG8=", mimeType: "image/png" };

describe("openAIToAgent (no originals — stock-compatible path)", () => {
  it("emits toolResult content as blocks so transports can safely filter", () => {
    const messages: OpenAIMessage[] = [
      { role: "tool", content: "tool output", tool_call_id: "call_123" },
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

  it("rebuilds assistant toolCall blocks from OpenAI tool_calls", () => {
    const result = openAIToAgent([
      {
        role: "assistant",
        content: "Running.",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "exec", arguments: '{"command":"ls"}' } },
        ],
      },
    ]);
    expect(result[0].content).toEqual([
      { type: "text", text: "Running." },
      { type: "toolCall", id: "call_1", name: "exec", arguments: { command: "ls" } },
    ]);
  });

  it("does not leak internal wire hints (hrIndex) into AgentMessages", () => {
    const result = openAIToAgent(agentToOpenAI([{ role: "user", content: "hi", timestamp: 5 }]));
    expect(result[0]).toEqual({ role: "user", content: "hi", timestamp: 5 });
  });
});

describe("normalizeAgentMessages", () => {
  it("normalizes assistant string content into OpenClaw blocks", () => {
    const result = normalizeAgentMessages([{ role: "assistant", content: "hello from headroom" }]);

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
    const result = normalizeAgentMessages([{ role: "toolResult", content: "tool output" }]);

    expect(result[0]).toMatchObject({
      role: "toolResult",
      content: [{ type: "text", text: "tool output" }],
      toolCallId: "unknown",
      tool_use_id: "unknown",
      toolName: "headroom",
      isError: false,
    });
  });

  it("preserves image blocks in tool results", () => {
    const result = normalizeAgentMessages([
      { role: "toolResult", toolName: "view_image", content: [IMAGE_BLOCK] },
    ]);
    expect(result[0].content).toEqual([IMAGE_BLOCK]);
  });

  it("preserves unknown assistant block types instead of dropping them", () => {
    const customBlock = { type: "custom_provider_block", payload: { ok: true } };
    const result = normalizeAgentMessages([
      { role: "assistant", content: [{ type: "text", text: "hi" }, customBlock] },
    ]);
    expect(result[0].content).toEqual([{ type: "text", text: "hi" }, customBlock]);
  });
});

describe("agentToOpenAI (wire format)", () => {
  it("captures assistant metadata needed for OpenClaw round-trips", () => {
    const result = agentToOpenAI([
      {
        role: "assistant",
        content: "hello",
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-sonnet-4-5",
        stopReason: "stop",
      },
    ]);

    expect(result[0]._headroomMeta).toMatchObject({
      api: "anthropic-messages",
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      stopReason: "stop",
      hrIndex: 0,
    });
  });

  it("includes tool name on OpenAI tool messages for proxy protect-list matching", () => {
    const result = agentToOpenAI([
      {
        role: "toolResult",
        toolCallId: "call_browser_1",
        toolName: "browser",
        content: [{ type: "text", text: "screenshot saved" }],
      },
    ]);

    expect(result[0]).toMatchObject({
      role: "tool",
      name: "browser",
      tool_call_id: "call_browser_1",
      content: "screenshot saved",
    });
    expect(result[0]._headroomMeta?.toolName).toBe("browser");
  });

  it("never puts image bytes on the wire — placeholder in content, no blocks in meta", () => {
    const bigImage = { type: "image", data: "A".repeat(40_000), mimeType: "image/png" };
    const result = agentToOpenAI([
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "view_image",
        content: [{ type: "text", text: "Loaded 1 image." }, bigImage],
      },
    ]);

    expect(result[0].content).toBe(
      "Loaded 1 image.\n[headroom-omitted image image/png 30000 bytes]",
    );
    expect(JSON.stringify(result[0])).not.toContain("AAAAAAAA");
    expect(result[0].name).toBe("view_image");
  });

  it("keeps thinking blocks off the wire (restored locally, never compressed)", () => {
    const result = agentToOpenAI([
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "secret plan" },
          { type: "text", text: "visible" },
        ],
      },
    ]);
    expect(result[0].content).toBe("visible");
    expect(JSON.stringify(result[0])).not.toContain("secret plan");
  });

  it("serializes Anthropic-shaped user tool_result blocks as text + placeholder", () => {
    const result = agentToOpenAI([
      {
        role: "user",
        content: [
          { type: "text", text: "here is the result" },
          { type: "tool_result", tool_use_id: "call_1", content: [{ type: "text", text: "payload" }] },
        ],
      },
    ]);
    expect(result[0].content).toBe("here is the result\n[headroom-omitted tool_result]");
  });
});

describe("openAIToAgent with originals — restore regardless of proxy echo", () => {
  function lossy(messages: OpenAIMessage[], dropMeta: boolean): OpenAIMessage[] {
    return messages.map((msg) => {
      const next: OpenAIMessage = { ...msg, content: msg.content ? "[compressed]" : msg.content };
      if (dropMeta) delete next._headroomMeta;
      return next;
    });
  }

  const toolOriginal = [
    {
      role: "toolResult",
      toolCallId: "call_img",
      toolName: "view_image",
      isError: false,
      timestamp: 42,
      content: [{ type: "text", text: "Loaded 1 image." }, IMAGE_BLOCK],
    },
  ];

  it.each([
    ["meta echoed", false],
    ["meta dropped by proxy", true],
  ])("restores image bytes, toolName, isError and timestamp (%s)", (_label, dropMeta) => {
    const wire = lossy(agentToOpenAI(toolOriginal), dropMeta);
    const restored = openAIToAgent(wire, { originals: toolOriginal });

    expect(restored[0]).toMatchObject({
      role: "toolResult",
      toolName: "view_image",
      toolCallId: "call_img",
      isError: false,
      timestamp: 42,
      content: [{ type: "text", text: "[compressed]" }, IMAGE_BLOCK],
    });
  });

  it("restores isError=true from the original even when the proxy rewrote the error text", () => {
    const original = [
      {
        role: "toolResult",
        toolCallId: "call_err",
        toolName: "view_image",
        isError: true,
        content: [{ type: "text", text: JSON.stringify({ status: "error", tool: "view_image" }) }],
      },
    ];
    const wire = lossy(agentToOpenAI(original), true);
    expect(openAIToAgent(wire, { originals: original })[0].isError).toBe(true);
  });

  it("matches tool results by tool_call_id even when the proxy dropped other messages", () => {
    const originals = [
      { role: "user", content: "a" },
      { role: "user", content: "b" },
      toolOriginal[0],
    ];
    const wire = agentToOpenAI(originals);
    const shortened = lossy([wire[2]], true); // rolling window dropped both users
    const restored = openAIToAgent(shortened, { originals });
    expect(restored).toHaveLength(1);
    expect(restored[0].content).toEqual([{ type: "text", text: "[compressed]" }, IMAGE_BLOCK]);
  });

  it("returns original blocks untouched when the proxy left the text unchanged", () => {
    const wire = agentToOpenAI(toolOriginal);
    const restored = openAIToAgent(wire, { originals: toolOriginal });
    expect(restored[0].content).toEqual(toolOriginal[0].content);
  });

  it("restores thinking + toolCall blocks on the assistant and does not duplicate text", () => {
    const original = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "planning screenshot" },
          { type: "text", text: "I'll capture " },
          { type: "toolCall", id: "call_browser_1", name: "browser", arguments: { action: "screenshot" } },
          { type: "text", text: "the screen." },
        ],
        api: "anthropic-messages",
        provider: "anthropic",
        model: "claude-sonnet-4-5",
        stopReason: "toolUse",
      },
    ];

    const unchanged = openAIToAgent(agentToOpenAI(original), { originals: original });
    expect(unchanged[0].content).toEqual(original[0].content);
    expect(unchanged[0]).toMatchObject({ provider: "anthropic", stopReason: "toolUse" });

    const compressed = openAIToAgent(lossy(agentToOpenAI(original), true), { originals: original });
    expect(compressed[0].content).toEqual([
      { type: "thinking", thinking: "planning screenshot" },
      { type: "text", text: "[compressed]" },
      { type: "toolCall", id: "call_browser_1", name: "browser", arguments: { action: "screenshot" } },
    ]);
  });

  it("restores user tool_result blocks and drops echoed placeholders", () => {
    const blocks = [
      { type: "text", text: "summary" },
      { type: "tool_result", tool_use_id: "call_1", content: [{ type: "text", text: "payload" }] },
    ];
    const original = [{ role: "user", content: blocks }];
    const wire = agentToOpenAI(original);

    expect(openAIToAgent(wire, { originals: original })[0].content).toEqual(blocks);

    const echoedPlaceholder: OpenAIMessage[] = [
      { ...wire[0], content: "shorter\n[headroom-omitted tool_result]" },
    ];
    expect(openAIToAgent(echoedPlaceholder, { originals: original })[0].content).toEqual([
      { type: "text", text: "shorter" },
      blocks[1],
    ]);
  });

  it("keeps plain string user content as a string", () => {
    const original = [{ role: "user", content: "hello" }];
    const restored = openAIToAgent(agentToOpenAI(original), { originals: original });
    expect(restored[0].content).toBe("hello");
  });
});

describe("deferred tool_call wrappers and protectToolResults", () => {
  const VIDEO_ID = "mcp:openrouter-video-vision:openrouter-video-vision__analyze_video";
  const VIDEO_TEXT = "t=0.5s the walker moves; t=1.5s it pauses at the gate; t=2.5s it resumes.";
  const PIXEL_TEXT = "[{\"id\":\"char_1\"},{\"id\":\"char_2\"}]";

  const originals = [
    { role: "user", content: "check the clip and list characters" },
    {
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "call_video",
          name: "tool_call",
          arguments: { id: VIDEO_ID, args: { video_path: "/tmp/c.mp4", prompt: "bugs?" } },
        },
        {
          type: "toolCall",
          id: "call_pixel",
          name: "tool_call",
          arguments: JSON.stringify({ id: "mcp:pixellab:pixellab__list_characters", args: {} }),
        },
        { type: "toolCall", id: "call_exec", name: "exec", arguments: { command: "ls" } },
      ],
    },
    { role: "toolResult", toolCallId: "call_video", toolName: "tool_call", content: VIDEO_TEXT },
    { role: "toolResult", toolCallId: "call_pixel", toolName: "tool_call", content: PIXEL_TEXT },
    { role: "toolResult", toolCallId: "call_exec", toolName: "exec", content: "a\nb\n" },
  ];

  it("sends resolved tool names on the wire instead of the tool_call wrapper", () => {
    const wire = agentToOpenAI(originals);
    const assistant = wire[1];
    expect(assistant.tool_calls?.map((tc: { function: { name: string } }) => tc.function.name)).toEqual([
      "mcp__openrouter-video-vision__analyze_video",
      "mcp__pixellab__list_characters",
      "exec",
    ]);
    expect(wire[2]).toMatchObject({ role: "tool", tool_call_id: "call_video", name: "mcp__openrouter-video-vision__analyze_video" });
    expect(wire[3]).toMatchObject({ role: "tool", tool_call_id: "call_pixel", name: "mcp__pixellab__list_characters" });
    expect(wire[4]).toMatchObject({ role: "tool", tool_call_id: "call_exec", name: "exec" });
    // Wrapper arguments are passed through unchanged so the proxy can still see them.
    expect(JSON.parse(assistant.tool_calls?.[0].function.arguments)).toEqual({
      id: VIDEO_ID,
      args: { video_path: "/tmp/c.mp4", prompt: "bugs?" },
    });
  });

  it("restores the wrapper name and original toolCall blocks after the round trip", () => {
    const wire = agentToOpenAI(originals);
    const restored = openAIToAgent(wire, { originals });
    const assistant = restored[1] as { content: Array<{ type: string; name?: string; id?: string }> };
    expect(assistant.content.filter((b) => b.type === "toolCall").map((b) => b.name)).toEqual([
      "tool_call",
      "tool_call",
      "exec",
    ]);
    expect(restored[2]).toMatchObject({ role: "toolResult", toolName: "tool_call", toolCallId: "call_video" });
  });

  function compressedByProxy(wire: OpenAIMessage[]): OpenAIMessage[] {
    return wire.map((msg) =>
      msg.role === "tool" ? { ...msg, content: `[compressed ${msg.tool_call_id}]`, _headroomMeta: undefined } : msg,
    );
  }

  it("restores protected tool results verbatim and leaves others compressed", () => {
    const wire = compressedByProxy(agentToOpenAI(originals));
    const restored = openAIToAgent(wire, {
      originals,
      protectedToolNames: new Set(["analyze_video"]),
    });

    expect(restored[2]).toMatchObject({
      toolCallId: "call_video",
      content: [{ type: "text", text: VIDEO_TEXT }],
    });
    expect(restored[3]).toMatchObject({
      toolCallId: "call_pixel",
      content: [{ type: "text", text: "[compressed call_pixel]" }],
    });
    expect(restored[4]).toMatchObject({
      toolCallId: "call_exec",
      content: [{ type: "text", text: "[compressed call_exec]" }],
    });
  });

  it("matches protected names by canonical, server-prefixed and glob spellings", () => {
    const wire = compressedByProxy(agentToOpenAI(originals));
    for (const entry of [
      "mcp__openrouter-video-vision__analyze_video",
      "openrouter-video-vision__analyze_video",
      "mcp__openrouter-video-vision__*",
    ]) {
      const restored = openAIToAgent(wire, { originals, protectedToolNames: new Set([entry]) });
      expect(restored[2]).toMatchObject({ content: [{ type: "text", text: VIDEO_TEXT }] });
      expect(restored[3]).toMatchObject({ content: [{ type: "text", text: "[compressed call_pixel]" }] });
    }
  });

  it("protects plain (non-wrapped) tools by their own name", () => {
    const wire = compressedByProxy(agentToOpenAI(originals));
    const restored = openAIToAgent(wire, { originals, protectedToolNames: new Set(["exec"]) });
    expect(restored[4]).toMatchObject({ content: [{ type: "text", text: "a\nb\n" }] });
    expect(restored[2]).toMatchObject({ content: [{ type: "text", text: "[compressed call_video]" }] });
  });

  it("falls back to the result's own toolName when no matching toolCall block exists", () => {
    const orphan = [{ role: "toolResult", toolCallId: "call_x", toolName: "browser", content: "page" }];
    const wire = compressedByProxy(agentToOpenAI(orphan));
    const restored = openAIToAgent(wire, { originals: orphan, protectedToolNames: new Set(["browser"]) });
    expect(restored[0]).toMatchObject({ content: [{ type: "text", text: "page" }] });
  });

  it("does nothing without originals or with an empty protected set", () => {
    const wire = compressedByProxy(agentToOpenAI(originals));
    expect(openAIToAgent(wire, { protectedToolNames: new Set(["analyze_video"]) })[2]).toMatchObject({
      content: [{ type: "text", text: "[compressed call_video]" }],
    });
    expect(openAIToAgent(wire, { originals, protectedToolNames: new Set() })[2]).toMatchObject({
      content: [{ type: "text", text: "[compressed call_video]" }],
    });
  });
});

describe("inferToolResultIsError fallback (no originals, no meta)", () => {
  it("flags the OpenClaw error envelope", () => {
    const result = openAIToAgent([
      {
        role: "tool",
        content: JSON.stringify({ status: "error", tool: "view_image", error: "denied" }),
        tool_call_id: "call_err",
        name: "view_image",
      },
    ]);
    expect(result[0].isError).toBe(true);
  });

  it("does not flag ordinary JSON tool output that merely contains status=error", () => {
    const result = openAIToAgent([
      {
        role: "tool",
        content: JSON.stringify({ status: "error", code: 503, body: "upstream said no" }),
        tool_call_id: "call_ok",
        name: "web_fetch",
      },
    ]);
    expect(result[0].isError).toBe(false);
  });
});

describe("estimateRoughTokens", () => {
  it("charges a fixed vision cost per image instead of base64 length", () => {
    const huge = { type: "image", data: "A".repeat(400_000), mimeType: "image/png" };
    const tokens = estimateRoughTokens([{ role: "toolResult", content: [huge] }]);
    expect(tokens).toBe(IMAGE_TOKEN_ESTIMATE);
  });
});
