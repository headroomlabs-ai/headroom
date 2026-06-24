/**
 * Tests for SharedContext — matches Python test_shared_context.py patterns.
 * Uses mocked proxy (no real server needed).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SharedContext } from "../src/shared-context.js";
import type { ContextEntry, SharedContextStats } from "../src/shared-context.js";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockCompressResponse(
  content: string,
  tokensBefore = 100,
  tokensAfter = 30,
) {
  return {
    ok: true,
    json: async () => ({
      messages: [{ role: "user", content: `[compressed] ${content.slice(0, 20)}...` }],
      tokens_before: tokensBefore,
      tokens_after: tokensAfter,
      tokens_saved: tokensBefore - tokensAfter,
      compression_ratio: tokensAfter / tokensBefore,
      transforms_applied: ["smart_crusher"],
      ccr_hashes: [],
    }),
  };
}

describe("SharedContext put/get", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("put and get compressed content", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("test data"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    const entry = await ctx.put("key1", "test data");

    expect(entry.key).toBe("key1");
    expect(entry.originalTokens).toBe(100);
    expect(entry.compressedTokens).toBe(30);
    expect(entry.savingsPercent).toBe(70);

    const result = ctx.get("key1");
    expect(result).toContain("[compressed]");
  });

  it("get with full=true returns original", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("full data"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    await ctx.put("key1", "full data");

    const full = ctx.get("key1", { full: true });
    expect(full).toBe("full data");
  });

  it("get missing key returns null", () => {
    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    expect(ctx.get("nonexistent")).toBeNull();
  });

  it("overwrite key replaces entry", async () => {
    mockFetch
      .mockResolvedValueOnce(mockCompressResponse("first"))
      .mockResolvedValueOnce(mockCompressResponse("second"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    await ctx.put("key1", "first");
    await ctx.put("key1", "second");

    const full = ctx.get("key1", { full: true });
    expect(full).toBe("second");
  });

  it("getEntry returns metadata", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("metadata test"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    await ctx.put("key1", "metadata test", { agent: "agent-A" });

    const entry = ctx.getEntry("key1");
    expect(entry).not.toBeNull();
    expect(entry!.agent).toBe("agent-A");
    expect(entry!.transforms).toEqual(["smart_crusher"]);
  });

  it("getEntry returns null for missing key", () => {
    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    expect(ctx.getEntry("missing")).toBeNull();
  });
});

describe("SharedContext expiry", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("expired entry returns null on get", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("expiring"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787", ttl: 1 });
    await ctx.put("key1", "expiring");

    // Manually expire by manipulating timestamp
    const entry = ctx.getEntry("key1")!;
    (entry as any).timestamp = Date.now() / 1000 - 10; // 10 seconds ago
    // Re-set entry with expired timestamp via internal map
    (ctx as any).entries.set("key1", entry);

    expect(ctx.get("key1")).toBeNull();
  });

  it("expired entry is cleaned from getEntry", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("expiring"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787", ttl: 1 });
    await ctx.put("key1", "expiring");

    const entry = ctx.getEntry("key1")!;
    (entry as any).timestamp = Date.now() / 1000 - 10;
    (ctx as any).entries.set("key1", entry);

    expect(ctx.getEntry("key1")).toBeNull();
  });
});

describe("SharedContext keys", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists active keys", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("data"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    await ctx.put("a", "data");
    await ctx.put("b", "data");

    expect(ctx.keys()).toEqual(["a", "b"]);
  });

  it("excludes expired keys", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("data"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787", ttl: 1 });
    await ctx.put("fresh", "data");
    await ctx.put("stale", "data");

    // Expire "stale"
    const entry = (ctx as any).entries.get("stale");
    entry.timestamp = Date.now() / 1000 - 10;

    expect(ctx.keys()).toEqual(["fresh"]);
  });
});

describe("SharedContext stats", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("aggregates statistics", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("data", 200, 60));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    await ctx.put("a", "data");
    await ctx.put("b", "data");

    const stats = ctx.stats();
    expect(stats.entries).toBe(2);
    expect(stats.totalOriginalTokens).toBe(400);
    expect(stats.totalCompressedTokens).toBe(120);
    expect(stats.totalTokensSaved).toBe(280);
    expect(stats.savingsPercent).toBe(70);
  });

  it("empty stats", () => {
    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    const stats = ctx.stats();
    expect(stats.entries).toBe(0);
    expect(stats.savingsPercent).toBe(0);
  });
});

describe("SharedContext eviction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("evicts oldest at capacity", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("data"));

    const ctx = new SharedContext({
      baseUrl: "http://test:8787",
      maxEntries: 2,
    });
    await ctx.put("first", "data");
    await ctx.put("second", "data");
    await ctx.put("third", "data"); // should evict "first"

    expect(ctx.keys()).toEqual(["second", "third"]);
    expect(ctx.get("first")).toBeNull();
  });
});

describe("SharedContext clear", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("removes all entries", async () => {
    mockFetch.mockResolvedValue(mockCompressResponse("data"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    await ctx.put("a", "data");
    await ctx.put("b", "data");

    ctx.clear();
    expect(ctx.keys()).toEqual([]);
    expect(ctx.stats().entries).toBe(0);
  });
});

describe("SharedContext fallback on proxy failure", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("stores uncompressed when proxy unavailable", async () => {
    mockFetch.mockRejectedValue(new Error("connection refused"));

    const ctx = new SharedContext({ baseUrl: "http://test:8787" });
    const entry = await ctx.put("key1", "fallback data");

    expect(entry.key).toBe("key1");
    expect(entry.original).toBe("fallback data");
    // Uncompressed — original equals compressed
    expect(entry.compressed).toBe("fallback data");
  });
});
