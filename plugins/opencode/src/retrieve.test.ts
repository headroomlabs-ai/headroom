import { afterEach, describe, expect, it, vi } from "vitest";

import { createHeadroomRetrieveTool, trimTrailingSlashes } from "./retrieve.js";

afterEach(() => vi.unstubAllGlobals());

describe("retrieve supported CCR identifiers", () => {
  it.each(["ab", "ABC", "0123456789abcdef01234567"])("retrieves %s", async (hash) => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify("original content")));
    vi.stubGlobal("fetch", fetchMock);
    const tool = createHeadroomRetrieveTool({ proxyBaseUrl: "http://127.0.0.1:8787" });
    expect(await tool.execute({ hash })).toBe("original content");
    expect(fetchMock).toHaveBeenCalledWith(
      `http://127.0.0.1:8787/v1/retrieve/${hash}`,
      expect.any(Object),
    );
  });

  it.each(["a", "a".repeat(25), "zz", "ab cd", "ab?key=x"])("rejects %s", async (hash) => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const tool = createHeadroomRetrieveTool({ proxyBaseUrl: "http://127.0.0.1:8787" });
    expect(JSON.parse(await tool.execute({ hash })).error).toContain("Invalid hash format");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

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
