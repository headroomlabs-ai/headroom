import { describe, it, expect, vi, beforeEach } from "vitest";
import { compress } from "../src/compress.js";
import type { OpenAIMessage } from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function okResponse(overrides = {}) {
  return new Response(
    JSON.stringify({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 100,
      tokens_after: 30,
      tokens_saved: 70,
      compression_ratio: 0.3,
      transforms_applied: ["smart_crusher"],
      ccr_hashes: [],
      ...overrides,
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

describe("compress()", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("compresses messages and returns CompressResult", async () => {
    mockFetch.mockResolvedValueOnce(okResponse());

    const messages: OpenAIMessage[] = [{ role: "user", content: "hello" }];
    const result = await compress(messages, {
      model: "gpt-4o",
      baseUrl: "http://localhost:8787",
    });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBe(100);
    expect(result.tokensAfter).toBe(30);
    expect(result.tokensSaved).toBe(70);
  });

  it("passes all options through to the HTTP call", async () => {
    mockFetch.mockResolvedValueOnce(okResponse());

    await compress([{ role: "user", content: "hello" }], {
      model: "claude-sonnet-4-5-20250929",
      baseUrl: "http://custom:9999",
      apiKey: "hr_mykey",
    });

    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe("http://custom:9999/v1/compress");
    expect(opts.headers["Authorization"]).toBe("Bearer hr_mykey");
    const body = JSON.parse(opts.body);
    expect(body.model).toBe("claude-sonnet-4-5-20250929");
  });

  it("uses provided client instance instead of creating new one", async () => {
    const mockClient = {
      compress: vi.fn().mockResolvedValue({
        messages: [],
        tokensBefore: 0,
        tokensAfter: 0,
        tokensSaved: 0,
        compressionRatio: 1.0,
        transformsApplied: [],
        ccrHashes: [],
        compressed: false,
      }),
    };

    await compress([{ role: "user", content: "hello" }], {
      client: mockClient,
      model: "gpt-4o",
    });

    expect(mockClient.compress).toHaveBeenCalledOnce();
    expect(mockFetch).not.toHaveBeenCalled(); // no HTTP call
  });

  it("handles multi-turn conversation with tool calls", async () => {
    mockFetch.mockResolvedValueOnce(okResponse());

    const messages: OpenAIMessage[] = [
      { role: "system", content: "You are a helpful assistant" },
      { role: "user", content: "Search for data" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "tc_1",
            type: "function",
            function: { name: "search", arguments: '{"q":"data"}' },
          },
        ],
      },
      {
        role: "tool",
        content: '{"results": [1, 2, 3]}',
        tool_call_id: "tc_1",
      },
      { role: "user", content: "Summarize" },
    ];

    const result = await compress(messages, {
      model: "gpt-4o",
      baseUrl: "http://localhost:8787",
    });

    expect(result.compressed).toBe(true);
    // Verify all messages were sent to proxy
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.messages).toHaveLength(5);
  });
});
