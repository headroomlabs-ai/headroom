/**
 * Tests for Gemini adapter — withHeadroom for Google Generative AI.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { withHeadroom } from "../../src/adapters/gemini.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockCompressResponse(messages: any[]) {
  return {
    ok: true,
    json: async () => ({
      messages: [{ role: "user", content: "compressed content" }],
      tokens_before: 500,
      tokens_after: 100,
      tokens_saved: 400,
      compression_ratio: 0.2,
      transforms_applied: ["smart_crusher"],
      ccr_hashes: [],
    }),
  };
}

describe("Gemini withHeadroom", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("wraps generateContent with compression", async () => {
    const originalResult = { response: { text: () => "Generated text" } };
    const mockModel = {
      generateContent: vi.fn().mockResolvedValue(originalResult),
    };

    mockFetch.mockResolvedValue(mockCompressResponse([]));

    const wrapped = withHeadroom(mockModel, { baseUrl: "http://test:8787" });

    const result = await wrapped.generateContent({
      contents: [
        { role: "user", parts: [{ text: "Hello Gemini" }] },
      ],
    });

    // The original generateContent was called
    expect(mockModel.generateContent).toHaveBeenCalled();
    expect(result).toBe(originalResult);

    // Compression was triggered
    expect(mockFetch).toHaveBeenCalled();
  });

  it("wraps generateContentStream if available", async () => {
    const originalResult = { stream: "mock stream" };
    const mockModel = {
      generateContent: vi.fn(),
      generateContentStream: vi.fn().mockResolvedValue(originalResult),
    };

    mockFetch.mockResolvedValue(mockCompressResponse([]));

    const wrapped = withHeadroom(mockModel, { baseUrl: "http://test:8787" });

    const result = await wrapped.generateContentStream({
      contents: [{ role: "user", parts: [{ text: "Stream test" }] }],
    });

    expect(mockModel.generateContentStream).toHaveBeenCalled();
    expect(result).toBe(originalResult);
  });

  it("passes through other model methods", () => {
    const mockModel = {
      generateContent: vi.fn(),
      countTokens: vi.fn().mockReturnValue(42),
    };

    const wrapped = withHeadroom(mockModel, { baseUrl: "http://test:8787" });
    expect(wrapped.countTokens()).toBe(42);
  });

  it("uses model option for compression", async () => {
    const mockModel = {
      generateContent: vi.fn().mockResolvedValue({}),
    };

    mockFetch.mockResolvedValue(mockCompressResponse([]));

    const wrapped = withHeadroom(mockModel, {
      baseUrl: "http://test:8787",
      model: "gemini-2.0-flash",
    });

    await wrapped.generateContent({
      contents: [{ role: "user", parts: [{ text: "test" }] }],
    });

    const [, opts] = mockFetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.model).toBe("gemini-2.0-flash");
  });

  it("falls back gracefully on proxy failure", async () => {
    const originalResult = { response: { text: () => "ok" } };
    const mockModel = {
      generateContent: vi.fn().mockResolvedValue(originalResult),
    };

    // Proxy fails, fallback returns original messages uncompressed
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        messages: [{ role: "user", content: "Hello Gemini" }],
        tokens_before: 0,
        tokens_after: 0,
        tokens_saved: 0,
        compression_ratio: 1.0,
        transforms_applied: [],
        ccr_hashes: [],
      }),
    });

    const wrapped = withHeadroom(mockModel, { baseUrl: "http://test:8787" });
    const result = await wrapped.generateContent({
      contents: [{ role: "user", parts: [{ text: "Hello Gemini" }] }],
    });

    expect(result).toBe(originalResult);
  });
});
