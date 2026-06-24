/**
 * Tests for case conversion utilities.
 */
import { describe, it, expect } from "vitest";
import {
  snakeToCamel,
  camelToSnake,
  deepCamelCase,
  deepSnakeCase,
} from "../../src/utils/case.js";

describe("snakeToCamel", () => {
  it("converts simple snake_case", () => {
    expect(snakeToCamel("hello_world")).toBe("helloWorld");
  });

  it("converts multiple underscores", () => {
    expect(snakeToCamel("tokens_input_before")).toBe("tokensInputBefore");
  });

  it("handles already camelCase", () => {
    expect(snakeToCamel("alreadyCamel")).toBe("alreadyCamel");
  });

  it("handles single word", () => {
    expect(snakeToCamel("hello")).toBe("hello");
  });

  it("handles numbers", () => {
    expect(snakeToCamel("bm25_k1")).toBe("bm25K1");
  });
});

describe("camelToSnake", () => {
  it("converts camelCase to snake_case", () => {
    expect(camelToSnake("helloWorld")).toBe("hello_world");
  });

  it("converts multiple capitals", () => {
    expect(camelToSnake("tokensInputBefore")).toBe("tokens_input_before");
  });

  it("handles already snake_case", () => {
    expect(camelToSnake("already_snake")).toBe("already_snake");
  });
});

describe("deepCamelCase", () => {
  it("converts flat object", () => {
    const result = deepCamelCase({
      tokens_before: 100,
      tokens_after: 50,
    });
    expect(result).toEqual({ tokensBefore: 100, tokensAfter: 50 });
  });

  it("converts nested objects", () => {
    const result = deepCamelCase({
      cache_metrics: {
        stable_prefix_hash: "abc",
        prefix_changed: true,
      },
    });
    expect(result).toEqual({
      cacheMetrics: {
        stablePrefixHash: "abc",
        prefixChanged: true,
      },
    });
  });

  it("converts arrays", () => {
    const result = deepCamelCase([
      { tool_name: "search" },
      { tool_name: "read" },
    ]);
    expect(result).toEqual([
      { toolName: "search" },
      { toolName: "read" },
    ]);
  });

  it("handles null and undefined", () => {
    expect(deepCamelCase(null)).toBeNull();
    expect(deepCamelCase(undefined)).toBeUndefined();
  });

  it("handles primitive values", () => {
    expect(deepCamelCase(42)).toBe(42);
    expect(deepCamelCase("hello")).toBe("hello");
    expect(deepCamelCase(true)).toBe(true);
  });
});

describe("deepSnakeCase", () => {
  it("converts flat object", () => {
    const result = deepSnakeCase({
      tokensBefore: 100,
      tokensAfter: 50,
    });
    expect(result).toEqual({ tokens_before: 100, tokens_after: 50 });
  });

  it("converts nested objects", () => {
    const result = deepSnakeCase({
      cacheMetrics: {
        stablePrefixHash: "abc",
      },
    });
    expect(result).toEqual({
      cache_metrics: {
        stable_prefix_hash: "abc",
      },
    });
  });

  it("round-trips with deepCamelCase", () => {
    const original = {
      tokens_before: 100,
      cache_metrics: {
        stable_prefix_hash: "abc",
        items: [{ tool_name: "x" }],
      },
    };
    const camel = deepCamelCase(original);
    const back = deepSnakeCase(camel);
    expect(back).toEqual(original);
  });
});
