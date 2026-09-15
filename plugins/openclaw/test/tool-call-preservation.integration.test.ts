/**
 * Mock integration tests for tool-call preservation.
 *
 * Simulates proxy /v1/compress behavior (lossy tool text, optional CCR hashes)
 * through the real convert layer and HeadroomContextEngine.assemble().
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { agentToOpenAI, openAIToAgent, type OpenAIMessage } from "../src/convert.js";

const mocked = vi.hoisted(() => ({
  start: vi.fn(async () => "http://127.0.0.1:8787"),
  stop: vi.fn(async () => undefined),
  logger: {
    debug: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
  },
}));

vi.mock("headroom-ai", () => ({
  compress: vi.fn(),
}));

vi.mock("../src/openclaw-compaction.js", () => ({
  delegateCompactionToRuntime: vi.fn(),
}));

vi.mock("../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = mocked.start;
    stop = mocked.stop;
  },
  defaultLogger: mocked.logger,
}));

import { compress } from "headroom-ai";
import { HeadroomContextEngine } from "../src/engine.js";

afterEach(() => {
  vi.mocked(compress).mockReset();
});

function godotVisionTranscript() {
  return [
    { role: "user", content: "Capture the Godot editor screenshot and describe it.", timestamp: 1 },
    {
      role: "assistant",
      content: [
        { type: "text", text: "I'll grab a screenshot." },
        {
          type: "toolCall",
          id: "call_browser_1",
          name: "browser",
          arguments: { action: "screenshot" },
        },
      ],
      api: "anthropic-messages",
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      stopReason: "toolUse",
      timestamp: 2,
    },
    {
      role: "toolResult",
      toolCallId: "call_browser_1",
      toolName: "browser",
      content: [{ type: "text", text: "Screenshot saved to /tmp/godot.png" }],
      timestamp: 3,
    },
    {
      role: "assistant",
      content: [
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
      timestamp: 4,
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
      timestamp: 5,
    },
  ];
}

/**
 * Simulates proxy lossy-compressing tool message strings but returning OpenAI-shaped messages.
 * `dropMeta` mimics proxy transforms that rebuild message objects and drop `_headroomMeta`.
 */
function mockLossyCompress(options?: { ccrHashes?: string[]; dropMeta?: boolean }) {
  vi.mocked(compress).mockImplementation(async (messages) => {
    const openaiIn = messages as OpenAIMessage[];
    const compressed = openaiIn.map((msg) => {
      const next: OpenAIMessage = { ...msg };
      if (options?.dropMeta) delete next._headroomMeta;
      if (msg.role === "tool") {
        return { ...next, content: "[lossy summary of tool output]" };
      }
      if (msg.role === "assistant" && msg.content && msg.content.length > 200) {
        return { ...next, content: msg.content.slice(0, 200) + "…" };
      }
      return next;
    });

    return {
      compressed: true,
      messages: compressed,
      tokensBefore: 120_000,
      tokensAfter: 45_000,
      tokensSaved: 75_000,
      compressionRatio: 0.375,
      transformsApplied: ["ContentRouter", "SmartCrusher"],
      ccrHashes: options?.ccrHashes ?? [],
    };
  });
}

describe("tool-call preservation mock integration", () => {
  it("agentToOpenAI emits tool names and keeps image bytes off the wire", () => {
    const openai = agentToOpenAI(godotVisionTranscript());
    const viewImage = openai.find((msg) => msg.role === "tool" && msg.name === "view_image");
    const browser = openai.find((msg) => msg.role === "tool" && msg.name === "browser");

    expect(viewImage).toMatchObject({
      role: "tool",
      name: "view_image",
      tool_call_id: "call_view_1",
      content: "[headroom-omitted image image/png 33 bytes]",
    });
    expect(JSON.stringify(openai)).not.toContain("iVBORw0KGgo");
    expect(browser).toMatchObject({
      role: "tool",
      name: "browser",
      tool_call_id: "call_browser_1",
    });
  });

  it.each([
    ["meta echoed", false],
    ["meta dropped by proxy", true],
  ])("assemble() restores view_image bytes after mock lossy proxy compression (%s)", async (_label, dropMeta) => {
    mockLossyCompress({ dropMeta });

    const engine = new HeadroomContextEngine({
      assembleCompressConfig: { protect_recent: 2 },
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    const result = await engine.assemble({
      sessionId: "godot-dashboard",
      messages: godotVisionTranscript(),
    });

    const viewResult = result.messages.find(
      (msg) => msg.role === "toolResult" && msg.toolName === "view_image",
    );
    // The image-only tool result gets the lossy text folded in front of the image.
    expect(viewResult?.content).toEqual([
      { type: "text", text: "[lossy summary of tool output]" },
      {
        type: "image",
        data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
        mimeType: "image/png",
      },
    ]);
    expect(viewResult).toMatchObject({ toolCallId: "call_view_1", isError: false, timestamp: 5 });

    const assistants = result.messages.filter((msg) => msg.role === "assistant");
    expect(assistants.map((msg) => msg.provider)).toEqual(["anthropic", "anthropic"]);
    expect(result.systemPromptAddition).toBeUndefined();
  });

  it("assemble() includes CCR hint only when mock proxy returns hashes", async () => {
    mockLossyCompress({ ccrHashes: ["deadbeef"] });

    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    const result = await engine.assemble({
      sessionId: "godot-dashboard",
      messages: godotVisionTranscript(),
    });

    expect(result.systemPromptAddition).toContain("headroom_retrieve");
  });

  it("assemble() passes protect_recent in compress config to mocked SDK", async () => {
    mockLossyCompress();

    const engine = new HeadroomContextEngine({
      assembleCompressConfig: { protect_recent: 2 },
      skipAssembleWhenGatewayRouted: false,
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await engine.assemble({
      sessionId: "godot-dashboard",
      messages: godotVisionTranscript(),
    });

    expect(compress).toHaveBeenCalledWith(
      expect.any(Array),
      expect.objectContaining({
        config: { protect_recent: 2 },
      }),
    );

    const openaiPayload = (vi.mocked(compress).mock.calls[0]?.[0] ?? []) as OpenAIMessage[];
    expect(openaiPayload.some((msg) => msg.role === "tool" && msg.name === "view_image")).toBe(
      true,
    );
  });

  it("skipAssembleWhenGatewayRouted bypasses compress for a gateway-routed provider", async () => {
    const engine = new HeadroomContextEngine({
      skipAssembleWhenGatewayRouted: true,
      gatewayProviderIds: ["openrouter", "minimax-portal"],
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    const messages = godotVisionTranscript();
    const result = await engine.assemble({
      sessionId: "godot-dashboard",
      messages,
      runtimeSettings: { model: { provider: "openrouter" } },
    });

    expect(compress).not.toHaveBeenCalled();
    const viewResult = result.messages.find(
      (msg) => msg.role === "toolResult" && msg.toolName === "view_image",
    );
    expect(viewResult?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "image", mimeType: "image/png" }),
      ]),
    );
  });

  it("skipAssembleWhenGatewayRouted still compresses for a provider that is not routed", async () => {
    mockLossyCompress();
    const engine = new HeadroomContextEngine({
      skipAssembleWhenGatewayRouted: true,
      gatewayProviderIds: ["openrouter"],
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await engine.assemble({
      sessionId: "godot-dashboard",
      messages: godotVisionTranscript(),
      runtimeSettings: { model: { provider: "anthropic" } },
    });

    expect(compress).toHaveBeenCalledTimes(1);
  });

  it("full round-trip: OpenClaw → OpenAI → lossy OpenAI (meta stripped) → OpenClaw keeps toolCall + image", () => {
    const original = godotVisionTranscript();
    const openai = agentToOpenAI(original);

    const lossyOpenai = openai.map(({ _headroomMeta: _dropped, ...msg }) =>
      msg.role === "tool" ? { ...msg, content: "[lossy]" } : msg,
    );

    const restored = openAIToAgent(lossyOpenai, { originals: original });
    const viewResult = restored.find(
      (msg) => msg.role === "toolResult" && msg.toolName === "view_image",
    );
    const assistantWithBrowser = restored.find(
      (msg) =>
        msg.role === "assistant" &&
        Array.isArray(msg.content) &&
        msg.content.some((block: { name?: string }) => block?.name === "browser"),
    );

    expect(viewResult?.content).toEqual([
      { type: "text", text: "[lossy]" },
      expect.objectContaining({ type: "image", mimeType: "image/png" }),
    ]);
    expect(assistantWithBrowser?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "toolCall", name: "browser" }),
      ]),
    );
  });
});
