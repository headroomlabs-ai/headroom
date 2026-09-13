import { describe, expect, it } from "vitest";

import { PreparedCache } from "../src/cache.js";
import type { PreparedEntry } from "../src/types.js";

function entry(
  key: string,
  sizeBytes: number,
  hash = `hash-${key}`,
): PreparedEntry {
  return {
    key,
    toolCallId: `call-${key}`,
    toolName: "bash",
    originalText: `original-${key}`,
    compressedText: `compressed-${key}`,
    ccrHashes: [hash],
    retrievals: new Map([[hash, `original-for-${hash}`]]),
    tokensBefore: 1_000,
    tokensAfter: 500,
    tokensSaved: 500,
    originalBytes: Buffer.byteLength(`original-${key}`),
    compressedBytes: Buffer.byteLength(`compressed-${key}`),
    sizeBytes,
    originalSha256: `sha-${key}`,
    policyVersion: "v1",
    createdAt: 1,
    lastAccessedAt: 1,
  };
}

describe("PreparedCache", () => {
  it("accounts exact bytes and rejects an oversized entry", () => {
    const cache = new PreparedCache(10);

    expect(cache.set(entry("a", 6))).toBe(true);
    expect(cache.stats()).toEqual({ entries: 1, bytes: 6, maxBytes: 10 });
    expect(cache.set(entry("large", 11))).toBe(false);
    expect(cache.stats()).toEqual({ entries: 1, bytes: 6, maxBytes: 10 });
  });

  it("touches entries on get before LRU eviction", () => {
    const cache = new PreparedCache(10);
    cache.set(entry("a", 5));
    cache.set(entry("b", 5));

    expect(cache.get("a")?.key).toBe("a");
    cache.set(entry("c", 5));

    expect(cache.get("a")?.key).toBe("a");
    expect(cache.get("b")).toBeUndefined();
    expect(cache.get("c")?.key).toBe("c");
  });

  it("evicts prepared and retrieval state atomically", () => {
    const cache = new PreparedCache(10);
    cache.set(entry("a", 6, "shared"));
    expect(cache.getRetrieval("shared")).toBe("original-for-shared");

    cache.set(entry("b", 6, "other"));

    expect(cache.get("a")).toBeUndefined();
    expect(cache.getRetrieval("shared")).toBeUndefined();
    expect(cache.getRetrieval("other")).toBe("original-for-other");
  });

  it("does not delete a newer hash owner when evicting an older entry", () => {
    const cache = new PreparedCache(12);
    cache.set(entry("a", 6, "shared"));
    cache.set(entry("b", 6, "shared"));
    cache.set(entry("c", 6, "other"));

    expect(cache.getRetrieval("shared")).toBe("original-for-shared");
    expect(cache.get("b")?.key).toBe("b");
  });
});
