import { describe, expect, it } from "vitest";

import { trimTrailingSlashes } from "./retrieve.js";

describe("trimTrailingSlashes", () => {
  it.each([
    ["http://localhost:8787", "http://localhost:8787"],
    ["http://localhost:8787/", "http://localhost:8787"],
    ["http://localhost:8787///", "http://localhost:8787"],
    ["http://localhost:8787/v1/", "http://localhost:8787/v1"],
    ["///", ""],
    ["", ""],
  ])("trims %j to %j", (input, expected) => {
    expect(trimTrailingSlashes(input)).toBe(expected);
  });

  it("runs in linear time on long runs of slashes", () => {
    // The previous /\/+$/ regex backtracked quadratically on a long run of
    // "/" that is not at the end of the string (CodeQL js/polynomial-redos).
    const slashes = "/".repeat(100_000);
    const input = `http://localhost:8787${slashes}x${slashes}`;

    const start = performance.now();
    const result = trimTrailingSlashes(input);
    expect(performance.now() - start).toBeLessThan(1000);
    expect(result).toBe(`http://localhost:8787${slashes}x`);
  });
});
