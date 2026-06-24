/**
 * Tests for data models — validates TypeScript interfaces match Python dataclass shapes.
 * Matches Python test_config.py (Block, WasteSignals, TransformResult, RequestMetrics).
 */
import { describe, it, expect } from "vitest";
import type {
  WasteSignals,
  CachePrefixMetrics,
  TransformDiff,
  DiffArtifact,
  SimulationResult,
  RequestMetrics,
  Block,
  SessionStats,
  ValidationResult,
  MetricsSummary,
  HealthStatus,
  RetrieveResult,
  CCRStats,
  TelemetryStats,
  TOINPattern,
} from "../src/types/models.js";

describe("WasteSignals", () => {
  it("has all fields", () => {
    const ws: WasteSignals = {
      jsonBloatTokens: 100,
      htmlNoiseTokens: 50,
      base64Tokens: 200,
      whitespaceTokens: 30,
      dynamicDateTokens: 10,
      repetitionTokens: 20,
      total: 410,
    };
    expect(ws.total).toBe(410);
  });

  it("total with all zeros", () => {
    const ws: WasteSignals = {
      jsonBloatTokens: 0,
      htmlNoiseTokens: 0,
      base64Tokens: 0,
      whitespaceTokens: 0,
      dynamicDateTokens: 0,
      repetitionTokens: 0,
      total: 0,
    };
    expect(ws.total).toBe(0);
  });
});

describe("CachePrefixMetrics", () => {
  it("has all fields", () => {
    const m: CachePrefixMetrics = {
      stablePrefixBytes: 4096,
      stablePrefixTokensEst: 1024,
      stablePrefixHash: "abc123",
      prefixChanged: false,
    };
    expect(m.stablePrefixHash).toBe("abc123");
    expect(m.prefixChanged).toBe(false);
    expect(m.previousHash).toBeUndefined();
  });

  it("previousHash is optional", () => {
    const m: CachePrefixMetrics = {
      stablePrefixBytes: 0,
      stablePrefixTokensEst: 0,
      stablePrefixHash: "",
      prefixChanged: true,
      previousHash: "old_hash",
    };
    expect(m.previousHash).toBe("old_hash");
  });
});

describe("TransformDiff", () => {
  it("has all fields", () => {
    const d: TransformDiff = {
      transformName: "smart_crusher",
      tokensBefore: 1000,
      tokensAfter: 300,
      tokensSaved: 700,
      itemsRemoved: 15,
      itemsKept: 5,
      details: "Compressed 20 items to 5",
      durationMs: 12.5,
    };
    expect(d.tokensSaved).toBe(700);
    expect(d.durationMs).toBe(12.5);
  });
});

describe("DiffArtifact", () => {
  it("contains transforms list", () => {
    const a: DiffArtifact = {
      requestId: "req-123",
      originalTokens: 5000,
      optimizedTokens: 1500,
      totalTokensSaved: 3500,
      transforms: [
        {
          transformName: "smart_crusher",
          tokensBefore: 5000,
          tokensAfter: 1500,
          tokensSaved: 3500,
          itemsRemoved: 0,
          itemsKept: 0,
          details: "",
          durationMs: 0,
        },
      ],
    };
    expect(a.transforms).toHaveLength(1);
    expect(a.totalTokensSaved).toBe(3500);
  });
});

describe("Block", () => {
  it("supports all block kinds", () => {
    const kinds = [
      "system",
      "user",
      "assistant",
      "tool_call",
      "tool_result",
      "rag",
      "unknown",
    ] as const;

    for (const kind of kinds) {
      const b: Block = {
        kind,
        text: "content",
        tokensEst: 10,
        contentHash: "hash",
        sourceIndex: 0,
        flags: {},
      };
      expect(b.kind).toBe(kind);
    }
  });

  it("flags is a flexible record", () => {
    const b: Block = {
      kind: "tool_result",
      text: "data",
      tokensEst: 50,
      contentHash: "h1",
      sourceIndex: 3,
      flags: { is_error: true, tool_name: "search" },
    };
    expect(b.flags.is_error).toBe(true);
    expect(b.flags.tool_name).toBe("search");
  });
});

describe("RequestMetrics", () => {
  it("has all expected fields", () => {
    const m: RequestMetrics = {
      requestId: "r1",
      timestamp: "2025-01-01T00:00:00Z",
      model: "gpt-4o",
      stream: false,
      mode: "optimize",
      tokensInputBefore: 5000,
      tokensInputAfter: 1500,
      blockBreakdown: { user: 2, tool_result: 3 },
      wasteSignals: { json_bloat_tokens: 200 },
      stablePrefixHash: "abc",
      cacheAlignmentScore: 0.85,
      cacheableTokens: 1000,
      breakpointsInserted: 2,
      estimatedCacheHit: true,
      estimatedSavingsPercent: 70,
      semanticCacheHit: false,
      transformsApplied: ["smart_crusher"],
      toolUnitsDropped: 0,
      turnsDropped: 0,
      messagesHash: "hash123",
    };
    expect(m.tokensInputBefore).toBe(5000);
    expect(m.transformsApplied).toContain("smart_crusher");
  });

  it("optional fields can be null", () => {
    const m: RequestMetrics = {
      requestId: "r1",
      timestamp: "2025-01-01",
      model: "gpt-4o",
      stream: false,
      mode: "audit",
      tokensInputBefore: 100,
      tokensInputAfter: 100,
      tokensOutput: null,
      blockBreakdown: {},
      wasteSignals: {},
      stablePrefixHash: "",
      cacheAlignmentScore: 0,
      cachedTokens: null,
      cacheOptimizerUsed: null,
      cacheOptimizerStrategy: null,
      cacheableTokens: 0,
      breakpointsInserted: 0,
      estimatedCacheHit: false,
      estimatedSavingsPercent: 0,
      semanticCacheHit: false,
      transformsApplied: [],
      toolUnitsDropped: 0,
      turnsDropped: 0,
      messagesHash: "",
      error: null,
    };
    expect(m.tokensOutput).toBeNull();
    expect(m.error).toBeNull();
  });
});

describe("SimulationResult", () => {
  it("has all fields", () => {
    const s: SimulationResult = {
      tokensBefore: 5000,
      tokensAfter: 1500,
      tokensSaved: 3500,
      transforms: ["smart_crusher", "cache_aligner"],
      estimatedSavings: "70%",
      messagesOptimized: [],
      blockBreakdown: { user: 2 },
      wasteSignals: { json_bloat_tokens: 200 },
      stablePrefixHash: "abc",
      cacheAlignmentScore: 0.85,
    };
    expect(s.estimatedSavings).toBe("70%");
  });
});

describe("HealthStatus", () => {
  it("has all fields", () => {
    const h: HealthStatus = {
      status: "healthy",
      version: "0.5.18",
      config: { optimize: true, cache: true, rateLimit: false },
    };
    expect(h.status).toBe("healthy");
  });
});

describe("RetrieveResult", () => {
  it("has all fields", () => {
    const r: RetrieveResult = {
      hash: "abc123",
      originalContent: "data",
      originalTokens: 500,
      originalItemCount: 20,
      compressedItemCount: 5,
      toolName: "search",
      retrievalCount: 3,
    };
    expect(r.retrievalCount).toBe(3);
  });
});

describe("CCRStats", () => {
  it("has store and recentRetrievals", () => {
    const s: CCRStats = {
      store: {
        entries: 10,
        maxEntries: 1000,
        originalTokensCached: 50000,
        compressedTokensCached: 15000,
        retrievals: 5,
      },
      recentRetrievals: [
        {
          hash: "abc",
          query: null,
          itemsRetrieved: 20,
          totalItems: 100,
          toolName: "search",
          retrievalType: "full",
        },
      ],
    };
    expect(s.store.entries).toBe(10);
    expect(s.recentRetrievals).toHaveLength(1);
  });
});

describe("TelemetryStats", () => {
  it("has all fields", () => {
    const t: TelemetryStats = {
      enabled: true,
      totalCompressions: 500,
      totalRetrievals: 50,
      globalRetrievalRate: 0.1,
      toolSignaturesTracked: 25,
      avgCompressionRatio: 0.3,
      avgTokenReduction: 0.7,
    };
    expect(t.globalRetrievalRate).toBe(0.1);
  });
});

describe("TOINPattern", () => {
  it("has all fields", () => {
    const p: TOINPattern = {
      hash: "abc123def456",
      compressions: 100,
      retrievals: 10,
      retrievalRate: "10.0%",
      confidence: 0.95,
      skipRecommended: false,
      optimalMaxItems: 8,
    };
    expect(p.confidence).toBe(0.95);
    expect(p.skipRecommended).toBe(false);
  });
});
