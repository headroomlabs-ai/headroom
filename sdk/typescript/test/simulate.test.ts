/**
 * Tests for simulation API.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { simulate } from "../src/simulate.js";
import type { SimulationResult } from "../src/types/models.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockSimulateResponse() {
  return {
    ok: true,
    json: async () => ({
      tokens_before: 5000,
      tokens_after: 1500,
      tokens_saved: 3500,
      transforms: ["smart_crusher", "cache_aligner"],
      estimated_savings: "70%",
      messages_optimized: [{ role: "user", content: "compressed" }],
      block_breakdown: { user: 2, assistant: 1, tool_result: 3 },
      waste_signals: { json_bloat_tokens: 200, whitespace_tokens: 50 },
      stable_prefix_hash: "abc123",
      cache_alignment_score: 0.85,
      transforms_applied: ["smart_crusher"],
      ccr_hashes: [],
    }),
  };
}

describe("simulate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns SimulationResult with all fields", async () => {
    mockFetch.mockResolvedValue(mockSimulateResponse());

    const result = await simulate(
      [{ role: "user", content: "hello" }],
      { baseUrl: "http://test:8787", model: "gpt-4o" },
    );

    expect(result.tokensBefore).toBe(5000);
    expect(result.tokensAfter).toBe(1500);
    expect(result.tokensSaved).toBe(3500);
    expect(result.estimatedSavings).toBe("70%");
    expect(result.stablePrefixHash).toBe("abc123");
    expect(result.cacheAlignmentScore).toBe(0.85);
  });

  it("sends simulation config to proxy", async () => {
    mockFetch.mockResolvedValue(mockSimulateResponse());

    await simulate(
      [{ role: "user", content: "test" }],
      { baseUrl: "http://test:8787" },
    );

    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe("http://test:8787/v1/compress");
    const body = JSON.parse(opts.body);
    expect(body.config.default_mode).toBe("simulate");
    expect(body.config.generate_diff_artifact).toBe(true);
  });

  it("converts snake_case response to camelCase", async () => {
    mockFetch.mockResolvedValue(mockSimulateResponse());

    const result = await simulate(
      [{ role: "user", content: "test" }],
      { baseUrl: "http://test:8787" },
    );

    // Verify camelCase conversion
    expect(result).toHaveProperty("tokensBefore");
    expect(result).toHaveProperty("tokensAfter");
    expect(result).toHaveProperty("estimatedSavings");
    expect(result).toHaveProperty("blockBreakdown");
    expect(result).toHaveProperty("wasteSignals");
    expect(result).toHaveProperty("stablePrefixHash");
    expect(result).toHaveProperty("cacheAlignmentScore");
  });
});
