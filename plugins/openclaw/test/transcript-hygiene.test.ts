import { describe, expect, it } from "vitest";
import {
  resolveSoftThresholdTokens,
  resolveTranscriptHygieneSettings,
} from "../src/transcript-hygiene.js";

describe("resolveTranscriptHygieneSettings", () => {
  it("enables hygiene by default in hybrid mode", () => {
    expect(resolveTranscriptHygieneSettings({}, "hybrid")).toEqual({
      enabled: true,
      softThresholdTokens: 0,
      debounceMs: 30_000,
    });
  });

  it("disables hygiene by default in openclaw mode", () => {
    expect(resolveTranscriptHygieneSettings({}, "openclaw")).toEqual({
      enabled: false,
      softThresholdTokens: 0,
      debounceMs: 30_000,
    });
  });

  it("accepts explicit boolean and object overrides", () => {
    expect(resolveTranscriptHygieneSettings({ transcriptHygiene: false }, "hybrid")).toEqual({
      enabled: false,
      softThresholdTokens: 0,
      debounceMs: 30_000,
    });
    expect(
      resolveTranscriptHygieneSettings(
        { transcriptHygiene: { enabled: true, softThresholdTokens: 250_000, debounceMs: 5_000 } },
        "openclaw",
      ),
    ).toEqual({
      enabled: true,
      softThresholdTokens: 250_000,
      debounceMs: 5_000,
    });
  });
});

describe("resolveSoftThresholdTokens", () => {
  it("uses explicit soft threshold when set", () => {
    expect(resolveSoftThresholdTokens({ softThresholdTokens: 300_000 }, 1_000_000)).toBe(300_000);
  });

  it("defaults to half the token budget with a floor", () => {
    expect(resolveSoftThresholdTokens({}, 983_040)).toBe(491_520);
    expect(resolveSoftThresholdTokens({}, 120_000)).toBe(100_000);
  });

  it("falls back to 400k when budget is unknown", () => {
    expect(resolveSoftThresholdTokens({}, undefined)).toBe(400_000);
  });
});
