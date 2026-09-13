import { describe, expect, it } from "vitest";
import { HygieneDebounceTracker } from "../src/hygiene-debounce.js";

describe("HygieneDebounceTracker", () => {
  it("does not skip before any successful hygiene", () => {
    const tracker = new HygieneDebounceTracker();
    expect(tracker.shouldSkip("session-1", 30_000, 1_000)).toBe(false);
  });

  it("skips within the debounce window after a changed pass", () => {
    const tracker = new HygieneDebounceTracker();
    tracker.recordChanged("session-1", 1_000);
    expect(tracker.shouldSkip("session-1", 30_000, 5_000)).toBe(true);
    expect(tracker.shouldSkip("session-1", 30_000, 31_500)).toBe(false);
  });

  it("tracks sessions independently", () => {
    const tracker = new HygieneDebounceTracker();
    tracker.recordChanged("session-1", 1_000);
    expect(tracker.shouldSkip("session-2", 30_000, 5_000)).toBe(false);
  });
});
