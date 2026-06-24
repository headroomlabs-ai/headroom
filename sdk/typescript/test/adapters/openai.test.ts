import { describe, it, expect, vi, beforeEach } from "vitest";
import { withHeadroom } from "../../src/adapters/openai.js";

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

describe("withHeadroom (OpenAI)", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("intercepts chat.completions.create and compresses messages", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const mockCreate = vi.fn().mockResolvedValue({
      id: "chatcmpl-123",
      choices: [{ message: { role: "assistant", content: "hi" } }],
    });
    const fakeClient = {
      chat: { completions: { create: mockCreate } },
      embeddings: { create: vi.fn() },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "long message here" }],
    });

    // Should have called compress endpoint
    expect(mockFetch).toHaveBeenCalledOnce();
    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8787/v1/compress");

    // Should have called original create with compressed messages
    expect(mockCreate).toHaveBeenCalledOnce();
    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.messages).toEqual([
      { role: "user", content: "compressed" },
    ]);
    expect(createArgs.model).toBe("gpt-4o");
  });

  it("preserves non-chat methods unchanged", () => {
    const embeddingsCreate = vi.fn();
    const fakeClient = {
      chat: { completions: { create: vi.fn() } },
      embeddings: { create: embeddingsCreate },
      images: { generate: vi.fn() },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    expect(wrapped.embeddings.create).toBe(embeddingsCreate);
    expect(wrapped.images).toBe(fakeClient.images);
  });

  it("preserves return value from original create", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const expectedResponse = {
      id: "chatcmpl-123",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Hello!" },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 30, completion_tokens: 10, total_tokens: 40 },
    };
    const mockCreate = vi.fn().mockResolvedValue(expectedResponse);
    const fakeClient = {
      chat: { completions: { create: mockCreate } },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    const result = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hello" }],
    });

    expect(result).toEqual(expectedResponse);
  });

  it("passes model from params to compress", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const fakeClient = {
      chat: { completions: { create: vi.fn().mockResolvedValue({}) } },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
    });

    await wrapped.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hello" }],
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.model).toBe("gpt-4o-mini");
  });

  it("uses model from options when provided", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const fakeClient = {
      chat: { completions: { create: vi.fn().mockResolvedValue({}) } },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
      model: "gpt-4o",
    });

    await wrapped.chat.completions.create({
      model: "gpt-4o-mini", // this should be overridden for compression
      messages: [{ role: "user", content: "hello" }],
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.model).toBe("gpt-4o"); // from options
  });

  it("still passes original model to LLM even when compress model differs", async () => {
    mockFetch.mockResolvedValueOnce(mockCompressSuccess());

    const mockCreate = vi.fn().mockResolvedValue({});
    const fakeClient = {
      chat: { completions: { create: mockCreate } },
    };

    const wrapped = withHeadroom(fakeClient as any, {
      baseUrl: "http://localhost:8787",
      model: "gpt-4o", // compression model
    });

    await wrapped.chat.completions.create({
      model: "gpt-4o-mini", // LLM model
      messages: [{ role: "user", content: "hello" }],
    });

    // Original create should still get gpt-4o-mini
    const createArgs = mockCreate.mock.calls[0][0];
    expect(createArgs.model).toBe("gpt-4o-mini");
  });
});
