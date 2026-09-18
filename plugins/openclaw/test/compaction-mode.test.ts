import { describe, expect, it } from "vitest";
import {
  ownsPersistentCompaction,
  resolvePersistentCompactionMode,
} from "../src/compaction-mode.js";

describe("resolvePersistentCompactionMode", () => {
  it("defaults to openclaw (upstream-compatible delegation)", () => {
    expect(resolvePersistentCompactionMode({})).toBe("openclaw");
  });

  it("accepts explicit modes", () => {
    expect(resolvePersistentCompactionMode({ persistentCompaction: "openclaw" })).toBe("openclaw");
    expect(resolvePersistentCompactionMode({ persistentCompaction: "headroom" })).toBe("headroom");
    expect(resolvePersistentCompactionMode({ persistentCompaction: "hybrid" })).toBe("hybrid");
  });

  it("supports deprecated durableCompaction boolean alias", () => {
    expect(resolvePersistentCompactionMode({ durableCompaction: false })).toBe("openclaw");
    expect(resolvePersistentCompactionMode({ durableCompaction: true })).toBe("headroom");
  });

  it("maps ownsPersistentCompaction to ownsCompaction flag", () => {
    expect(ownsPersistentCompaction("headroom")).toBe(true);
    expect(ownsPersistentCompaction("openclaw")).toBe(false);
    expect(ownsPersistentCompaction("hybrid")).toBe(false);
  });
});
