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

  it("converts Anthropic content blocks to OpenAI format for compression", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const mockCreate = vi.fn().mockResolvedValue({});
    const fakeClient = { messages: { create: mockCreate } };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "analyze this data" }],
        },
      ],
      max_tokens: 1024,
    });

    // Verify the compress call used OpenAI format
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.messages[0].role).toBe("user");
    expect(body.messages[0].content).toBe("analyze this data");
  });

  it("handles Anthropic tool_use and tool_result blocks", async () => {
    // Proxy returns compressed messages
    mockFetch.mockResolvedValueOnce(
      mockCompressSuccess([
        { role: "user", content: "search for X" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "tu_1",
              type: "function",
              function: { name: "search", arguments: '{"q":"X"}' },
            },
          ],
        },
        {
          role: "tool",
          content: '{"results":["a","b"]}',
          tool_call_id: "tu_1",
        },
      ]),
    );

    const mockCreate = vi.fn().mockResolvedValue({});
    const fakeClient = { messages: { create: mockCreate } };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.messages.create({
      model: "claude-sonnet-4-5-20250929",
      messages: [
        { role: "user", content: "search for X" },
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "tu_1", name: "search", input: { q: "X" } },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tu_1",
              content: '{"results":["a","b"]}',
            },
          ],
        },
      ],
      max_tokens: 1024,
    });

    expect(mockCreate).toHaveBeenCalledOnce();
    // Messages should be converted back to Anthropic format
    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.messages.length).toBeGreaterThan(0);
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
