import { describe, it, expect, vi, beforeEach } from "vitest";
import { withHeadroom } from "../../src/adapters/anthropic.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockCompressSuccess(
  messages = [{ role: "user", content: "compressed" }],
) {
  return new Response(
    JSON.stringify({
      messages,
      tokens_before: 100,
      tokens_after: 30,
      tokens_saved: 70,
      compression_ratio: 0.3,
      transforms_applied: ["smart_crusher"],
      ccr_hashes: [],
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

// A thinking-enabled tool-use turn. Anthropic requires the thinking blocks
// back "complete and unmodified" with the tool result, or the request is a 400.
const thinkingToolTurn = [
  {
    role: "user",
    content: [
      { type: "text", text: "Read the gateway log.", cache_control: { type: "ephemeral" } },
    ],
  },
  {
    role: "assistant",
    content: [
      { type: "thinking", thinking: "I should read the log first.", signature: "SIG-abc123" },
      { type: "redacted_thinking", data: "REDACTED-xyz" },
      { type: "text", text: "Reading the log now." },
      { type: "tool_use", id: "toolu_01", name: "fetch_gateway_metrics", input: { path: "/var/log/gateway.json" } },
    ],
  },
  {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: "toolu_01",
        content: [
          { type: "text", text: "row 1\nrow 2\nrow 3" },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "AAAA" } },
        ],
      },
    ],
  },
];

describe("withHeadroom (Anthropic)", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("intercepts messages.create and compresses", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const mockCreate = vi.fn().mockResolvedValue({
      id: "msg_123",
      content: [{ type: "text", text: "response" }],
    });
    const fakeClient = {
      messages: { create: mockCreate },
      completions: { create: vi.fn() },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: [{ role: "user", content: "long message" }],
      max_tokens: 1024,
    });

    // Should have called compress
    expect(mockFetch).toHaveBeenCalledOnce();

    // Should have called original create
    expect(mockCreate).toHaveBeenCalledOnce();
    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.max_tokens).toBe(1024);
    expect(createArgs.model).toBe("claude-sonnet-4-5-20250929");
  });

  it("passes through non-messages methods", () => {
    const completionsCreate = vi.fn();
    const fakeClient = {
      messages: { create: vi.fn() },
      completions: { create: completionsCreate },
      beta: { something: vi.fn() },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    expect(wrapped.completions.create).toBe(completionsCreate);
    expect(wrapped.beta).toBe(fakeClient.beta);
  });

  it("preserves return value from original create", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const expectedResponse = {
      id: "msg_123",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Hello!" }],
      model: "claude-sonnet-4-5-20250929",
      stop_reason: "end_turn",
      usage: { input_tokens: 30, output_tokens: 10 },
    };
    const mockCreate = vi.fn().mockResolvedValue(expectedResponse);
    const fakeClient = { messages: { create: mockCreate } };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    const result = await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 1024,
    });

    expect(result).toEqual(expectedResponse);
  });

  it("posts Anthropic content blocks to /v1/compress unchanged", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const mockCreate = vi.fn().mockResolvedValue({});
    const fakeClient = { messages: { create: mockCreate } };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    const messages = [
      { role: "user", content: [{ type: "text", text: "analyze this data" }] },
    ];
    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages,
      max_tokens: 1024,
    });

    // The proxy compresses Anthropic blocks natively; nothing is converted.
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.messages).toEqual(messages);
    expect(body.model).toBe("claude-sonnet-4-5-20250929");
  });

  it("posts tool_use and array-content tool_result blocks as-is and forwards the proxy's result", async () => {
    const turn = [
      { role: "user", content: "run the query" },
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "tu_1", name: "query", input: { sql: "select 1" } },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "tu_1",
            content: [
              { type: "text", text: "row 1" },
              { type: "text", text: "row 2" },
            ],
          },
        ],
      },
    ];
    const compressed = structuredClone(turn);
    (compressed[2].content as any[])[0].content = [{ type: "text", text: "[2 rows]" }];
    mockFetch.mockResolvedValueOnce(mockCompressSuccess(compressed));

    const mockCreate = vi.fn().mockResolvedValue({});
    const fakeClient = { messages: { create: mockCreate } };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: turn,
      max_tokens: 1024,
    });

    // The proxy reads Anthropic tool_result blocks natively, list content included.
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.messages).toEqual(turn);

    expect(mockCreate).toHaveBeenCalledOnce();
    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.messages).toEqual(compressed);
  });

  it("keeps image blocks native in the compression request and in the forwarded messages", async () => {
    const turn = [
      {
        role: "user",
        content: [
          { type: "text", text: "what is this?" },
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "AAAA" },
          },
        ],
      },
    ];
    mockFetch.mockResolvedValueOnce(mockCompressSuccess(turn));

    const mockCreate = vi.fn().mockResolvedValue({});
    const fakeClient = { messages: { create: mockCreate } };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: turn,
      max_tokens: 1024,
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.messages).toEqual(turn);

    expect(mockCreate).toHaveBeenCalledOnce();
    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.messages).toEqual(turn);
  });

  it("posts a thinking-enabled tool-use turn unchanged, thinking blocks and tool_result blocks included", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess(thinkingToolTurn));
    const mockCreate = vi.fn().mockResolvedValue({});
    const wrapped = withHeadroom({ messages: { create: mockCreate } } as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: thinkingToolTurn,
      max_tokens: 1024,
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.messages).toEqual(thinkingToolTurn);
    expect(body.model).toBe("claude-sonnet-4-5-20250929");
  });

  it("forwards the proxy's returned Anthropic messages to messages.create as-is", async () => {
    const compressed = structuredClone(thinkingToolTurn);
    (compressed[2].content[0] as any).content[0].text = "[3 rows]";
    mockFetch.mockResolvedValueOnce(mockCompressSuccess(compressed));
    const mockCreate = vi.fn().mockResolvedValue({});
    const wrapped = withHeadroom({ messages: { create: mockCreate } } as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: thinkingToolTurn,
      max_tokens: 1024,
    });

    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.messages).toEqual(compressed);
    // The thinking blocks reach the provider exactly as the model produced them.
    expect(createArgs.messages[1].content.slice(0, 2)).toEqual(thinkingToolTurn[1].content.slice(0, 2));
  });

  it("returns original messages on compression fallback", async () => {
    // Simulate proxy unreachable
    mockFetch.mockRejectedValueOnce(new TypeError("fetch failed"));

    const originalMessages = [{ role: "user", content: "hello" }];
    const mockCreate = vi.fn().mockResolvedValue({});
    const fakeClient = { messages: { create: mockCreate } };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
      fallback: true,
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: originalMessages,
      max_tokens: 1024,
    });

    // Should still call original with uncompressed messages
    expect(mockCreate).toHaveBeenCalledOnce();
    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.messages).toEqual(originalMessages);
  });
});
