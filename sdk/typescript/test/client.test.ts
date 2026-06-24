import { describe, it, expect, vi, beforeEach } from "vitest";
import { HeadroomClient } from "../src/client.js";
import type { OpenAIMessage } from "../src/types.js";
import {
  HeadroomConnectionError,
  HeadroomAuthError,
  HeadroomCompressError,
} from "../src/types.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function okResponse(body: object) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function errorResponse(status: number, body: object) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const sampleMessages: OpenAIMessage[] = [
  { role: "user", content: "hello" },
  { role: "assistant", content: "hi there" },
];

const sampleProxyResponse = {
  messages: sampleMessages,
  tokens_before: 10,
  tokens_after: 8,
  tokens_saved: 2,
  compression_ratio: 0.8,
  transforms_applied: ["smart_crusher"],
  ccr_hashes: [],
};

describe("HeadroomClient", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("sends POST to /v1/compress with messages and model", async () => {
    mockFetch.mockResolvedValueOnce(okResponse(sampleProxyResponse));

    const client = new HeadroomClient({ baseUrl: "http://localhost:8787" });
    await client.compress(sampleMessages, { model: "gpt-4o" });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8787/v1/compress");
    expect(opts.method).toBe("POST");
    const body = JSON.parse(opts.body);
    expect(body.messages).toEqual(sampleMessages);
    expect(body.model).toBe("gpt-4o");
  });

  it("maps snake_case proxy response to camelCase result", async () => {
    mockFetch.mockResolvedValueOnce(
      okResponse({
        messages: [{ role: "user", content: "hello" }],
        tokens_before: 100,
        tokens_after: 30,
        tokens_saved: 70,
        compression_ratio: 0.3,
        transforms_applied: ["router:smart_crusher:0.35"],
        ccr_hashes: ["abc123"],
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://localhost:8787" });
    const result = await client.compress(sampleMessages, { model: "gpt-4o" });

    expect(result.tokensBefore).toBe(100);
    expect(result.tokensAfter).toBe(30);
    expect(result.tokensSaved).toBe(70);
    expect(result.compressionRatio).toBe(0.3);
    expect(result.transformsApplied).toEqual(["router:smart_crusher:0.35"]);
    expect(result.ccrHashes).toEqual(["abc123"]);
    expect(result.compressed).toBe(true);
  });

  it("sends apiKey as Authorization bearer header", async () => {
    mockFetch.mockResolvedValueOnce(okResponse(sampleProxyResponse));

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      apiKey: "hr_test123",
    });
    await client.compress(sampleMessages, { model: "gpt-4o" });

    const [, opts] = mockFetch.mock.calls[0];
    expect(opts.headers["Authorization"]).toBe("Bearer hr_test123");
  });

  it("does not send Authorization header when no apiKey", async () => {
    mockFetch.mockResolvedValueOnce(okResponse(sampleProxyResponse));

    const client = new HeadroomClient({ baseUrl: "http://localhost:8787" });
    await client.compress(sampleMessages, { model: "gpt-4o" });

    const [, opts] = mockFetch.mock.calls[0];
    expect(opts.headers["Authorization"]).toBeUndefined();
  });

  it("strips trailing slashes from baseUrl", async () => {
    mockFetch.mockResolvedValueOnce(okResponse(sampleProxyResponse));

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787///",
    });
    await client.compress(sampleMessages, { model: "gpt-4o" });

    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8787/v1/compress");
  });

  it("throws HeadroomAuthError on 401 (always, even with fallback)", async () => {
    mockFetch.mockResolvedValueOnce(
      errorResponse(401, {
        error: { type: "authentication_error", message: "Invalid API key" },
      }),
    );

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      fallback: true, // fallback doesn't apply to auth errors
    });

    await expect(
      client.compress(sampleMessages, { model: "gpt-4o" }),
    ).rejects.toThrow(HeadroomAuthError);
  });

  it("throws HeadroomCompressError on 400 (always, even with fallback)", async () => {
    mockFetch.mockResolvedValueOnce(
      errorResponse(400, {
        error: { type: "invalid_request", message: "Missing model" },
      }),
    );

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      fallback: true, // fallback doesn't apply to client errors
    });

    await expect(
      client.compress(sampleMessages, { model: "gpt-4o" }),
    ).rejects.toThrow(HeadroomCompressError);
  });

  it("falls back to uncompressed on network error when fallback=true", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("fetch failed"));

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      fallback: true,
    });
    const result = await client.compress(sampleMessages, { model: "gpt-4o" });

    expect(result.compressed).toBe(false);
    expect(result.messages).toEqual(sampleMessages);
    expect(result.tokensSaved).toBe(0);
  });

  it("throws HeadroomConnectionError on network error when fallback=false", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("fetch failed"));

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      fallback: false,
      retries: 0,
    });

    await expect(
      client.compress(sampleMessages, { model: "gpt-4o" }),
    ).rejects.toThrow(HeadroomConnectionError);
  });

  it("falls back on 503 when fallback=true", async () => {
    mockFetch.mockResolvedValueOnce(
      errorResponse(503, {
        error: { type: "compression_error", message: "Pipeline failed" },
      }),
    );

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      fallback: true,
      retries: 0,
    });
    const result = await client.compress(sampleMessages, { model: "gpt-4o" });

    expect(result.compressed).toBe(false);
    expect(result.messages).toEqual(sampleMessages);
  });

  it("retries on transient network errors then succeeds", async () => {
    mockFetch
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(okResponse(sampleProxyResponse));

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      retries: 1,
    });
    const result = await client.compress(sampleMessages, { model: "gpt-4o" });

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result.compressed).toBe(true);
  });

  it("retries on 503 then succeeds", async () => {
    mockFetch
      .mockResolvedValueOnce(
        errorResponse(503, {
          error: { type: "compression_error", message: "Busy" },
        }),
      )
      .mockResolvedValueOnce(okResponse(sampleProxyResponse));

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      retries: 1,
      fallback: false,
    });
    const result = await client.compress(sampleMessages, { model: "gpt-4o" });

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result.compressed).toBe(true);
  });

  it("does not retry on 400 client errors", async () => {
    mockFetch.mockResolvedValueOnce(
      errorResponse(400, {
        error: { type: "invalid_request", message: "Bad request" },
      }),
    );

    const client = new HeadroomClient({
      baseUrl: "http://localhost:8787",
      retries: 3,
      fallback: false,
    });

    await expect(
      client.compress(sampleMessages, { model: "gpt-4o" }),
    ).rejects.toThrow(HeadroomCompressError);

    expect(mockFetch).toHaveBeenCalledTimes(1); // no retries
  });

  it("defaults model to gpt-4o when not provided", async () => {
    mockFetch.mockResolvedValueOnce(okResponse(sampleProxyResponse));

    const client = new HeadroomClient({ baseUrl: "http://localhost:8787" });
    await client.compress(sampleMessages); // no model option

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.model).toBe("gpt-4o");
  });
});
