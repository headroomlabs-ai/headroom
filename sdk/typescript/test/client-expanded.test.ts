/**
 * Tests for expanded HeadroomClient — chat.completions, messages, metrics, CCR, etc.
 * Uses mocked proxy (no real server needed).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { HeadroomClient } from "../src/client.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(data: any, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: async () => data,
    text: async () => JSON.stringify(data),
  };
}

describe("HeadroomClient constructor", () => {
  it("accepts extended options", () => {
    const client = new HeadroomClient({
      baseUrl: "http://proxy:8787",
      apiKey: "hdr-key",
      providerApiKey: "sk-openai-key",
      defaultMode: "optimize",
      config: { smartCrusher: { enabled: true } },
    });
    expect(client).toBeDefined();
    expect(client.providerApiKey).toBe("sk-openai-key");
  });

  it("has chat.completions sub-client", () => {
    const client = new HeadroomClient();
    expect(client.chat).toBeDefined();
    expect(client.chat.completions).toBeDefined();
  });

  it("has messages sub-client", () => {
    const client = new HeadroomClient();
    expect(client.messages).toBeDefined();
  });

  it("has telemetry namespace", () => {
    const client = new HeadroomClient();
    expect(client.telemetry).toBeDefined();
    expect(client.telemetry.getStats).toBeTypeOf("function");
  });

  it("has feedback namespace", () => {
    const client = new HeadroomClient();
    expect(client.feedback).toBeDefined();
    expect(client.feedback.getStats).toBeTypeOf("function");
  });

  it("has toin namespace", () => {
    const client = new HeadroomClient();
    expect(client.toin).toBeDefined();
    expect(client.toin.getStats).toBeTypeOf("function");
  });
});

describe("HeadroomClient.health()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns health status", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        status: "healthy",
        version: "0.5.18",
        config: { optimize: true, cache: true, rate_limit: false },
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const health = await client.health();

    expect(health.status).toBe("healthy");
    expect(health.version).toBe("0.5.18");
    expect(health.config.optimize).toBe(true);
  });
});

describe("HeadroomClient.proxyStats()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns proxy stats with camelCase keys", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        requests: { total: 100, cached: 10, rate_limited: 0, failed: 2 },
        tokens: { input: 50000, output: 20000, saved: 30000 },
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const stats = await client.proxyStats();

    expect(stats.requests.total).toBe(100);
    expect(stats.requests.rateLimited).toBe(0);
    expect(stats.tokens.saved).toBe(30000);
  });
});

describe("HeadroomClient.getMetrics()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns request metrics from proxy", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        recent_requests: [
          { request_id: "r1", model: "gpt-4o", tokens_input_before: 1000, mode: "optimize" },
          { request_id: "r2", model: "claude", tokens_input_before: 2000, mode: "audit" },
        ],
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const metrics = await client.getMetrics();

    expect(metrics).toHaveLength(2);
    expect(metrics[0].requestId).toBe("r1");
    expect(metrics[0].tokensInputBefore).toBe(1000);
  });

  it("filters by model", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        recent_requests: [
          { request_id: "r1", model: "gpt-4o", mode: "optimize" },
          { request_id: "r2", model: "claude", mode: "optimize" },
        ],
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const metrics = await client.getMetrics({ model: "gpt-4o" });

    expect(metrics).toHaveLength(1);
    expect(metrics[0].requestId).toBe("r1");
  });

  it("applies limit", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        recent_requests: [
          { request_id: "r1" },
          { request_id: "r2" },
          { request_id: "r3" },
        ],
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const metrics = await client.getMetrics({ limit: 2 });

    expect(metrics).toHaveLength(2);
  });
});

describe("HeadroomClient.getSummary()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns aggregated summary", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        requests: { total: 50, by_model: { "gpt-4o": 30 }, failed: 1 },
        tokens: { total_before_compression: 100000, saved: 70000, savings_percent: 70 },
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const summary = await client.getSummary();

    expect(summary.totalRequests).toBe(50);
    expect(summary.totalTokensSaved).toBe(70000);
    expect(summary.errorCount).toBe(1);
  });
});

describe("HeadroomClient.validateSetup()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns valid when healthy", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        status: "healthy",
        config: { optimize: true },
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const result = await client.validateSetup();

    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
  });

  it("returns invalid when unhealthy", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        status: "unhealthy",
        config: {},
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const result = await client.validateSetup();

    expect(result.valid).toBe(false);
    expect(result.errors).toContain("Proxy unhealthy");
  });
});

describe("HeadroomClient.retrieve()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("retrieves by hash", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        hash: "abc123",
        original_content: "full data here",
        original_tokens: 500,
        original_item_count: 20,
        compressed_item_count: 5,
        tool_name: "search",
        retrieval_count: 1,
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const result = await client.retrieve("abc123") as any;

    expect(result.hash).toBe("abc123");
    expect(result.originalContent).toBe("full data here");
    expect(result.originalTokens).toBe(500);
  });

  it("retrieves with query", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        hash: "abc123",
        query: "error logs",
        results: [{ text: "ERROR: connection failed" }],
        count: 1,
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const result = await client.retrieve("abc123", { query: "error logs" }) as any;

    expect(result.query).toBe("error logs");
    expect(result.count).toBe(1);
  });
});

describe("HeadroomClient.clearCache()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("clears cache", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ status: "cleared" }));

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const result = await client.clearCache();

    expect(result.status).toBe("cleared");
  });
});

describe("HeadroomClient.close()", () => {
  it("is a no-op (HTTP client is stateless)", () => {
    const client = new HeadroomClient();
    expect(() => client.close()).not.toThrow();
  });
});

describe("HeadroomClient.compressRaw()", () => {
  beforeEach(() => vi.clearAllMocks());

  it("sends raw body to /v1/compress", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        messages: [],
        tokens_before: 100,
        tokens_after: 50,
      }),
    );

    const client = new HeadroomClient({ baseUrl: "http://test:8787" });
    const result = await client.compressRaw({
      messages: [{ role: "user", content: "test" }],
      model: "gpt-4o",
      config: { default_mode: "simulate" },
    });

    expect(result.tokens_before).toBe(100);

    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe("http://test:8787/v1/compress");
  });
});

describe("HeadroomClient config passthrough", () => {
  beforeEach(() => vi.clearAllMocks());

  it("sends config as snake_case in compress body", async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        messages: [{ role: "user", content: "hi" }],
        tokens_before: 10,
        tokens_after: 10,
        tokens_saved: 0,
        compression_ratio: 1.0,
        transforms_applied: [],
        ccr_hashes: [],
      }),
    );

    const client = new HeadroomClient({
      baseUrl: "http://test:8787",
      config: {
        smartCrusher: { enabled: true, minTokensToCrush: 100 },
      },
    });

    await client.compress([{ role: "user", content: "hi" }]);

    const [, opts] = mockFetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.config).toBeDefined();
    expect(body.config.smart_crusher.enabled).toBe(true);
    expect(body.config.smart_crusher.min_tokens_to_crush).toBe(100);
  });
});
