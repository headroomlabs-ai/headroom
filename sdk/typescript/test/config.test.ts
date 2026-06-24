/**
 * Tests for configuration types — matches Python test_config.py patterns.
 * Validates TypeScript interfaces match Python dataclass defaults and structure.
 */
import { describe, it, expect } from "vitest";
import type {
  HeadroomMode,
  HeadroomConfig,
  SmartCrusherConfig,
  ToolCrusherConfig,
  CacheAlignerConfig,
  RollingWindowConfig,
  IntelligentContextConfig,
  ScoringWeights,
  RelevanceScorerConfig,
  AnchorConfig,
  CacheOptimizerConfig,
  CCRConfig,
  PrefixFreezeConfig,
  ReadLifecycleConfig,
  CompressionProfile,
} from "../src/types/config.js";

describe("HeadroomMode", () => {
  it("accepts valid mode values", () => {
    const modes: HeadroomMode[] = ["audit", "optimize", "simulate"];
    expect(modes).toEqual(["audit", "optimize", "simulate"]);
  });
});

describe("ToolCrusherConfig", () => {
  it("accepts default-like values", () => {
    const config: ToolCrusherConfig = {
      enabled: false,
      minTokensToCrush: 500,
      maxArrayItems: 10,
      maxStringLength: 1000,
      maxDepth: 5,
    };
    expect(config.enabled).toBe(false);
    expect(config.minTokensToCrush).toBe(500);
    expect(config.maxArrayItems).toBe(10);
    expect(config.maxStringLength).toBe(1000);
    expect(config.maxDepth).toBe(5);
  });

  it("all fields are optional", () => {
    const config: ToolCrusherConfig = {};
    expect(config.enabled).toBeUndefined();
  });
});

describe("CacheAlignerConfig", () => {
  it("accepts default-like values", () => {
    const config: CacheAlignerConfig = {
      enabled: false,
      useDynamicDetector: true,
      entropyThreshold: 0.7,
      normalizeWhitespace: true,
      collapseBlankLines: true,
    };
    expect(config.enabled).toBe(false);
    expect(config.useDynamicDetector).toBe(true);
    expect(config.entropyThreshold).toBe(0.7);
  });
});

describe("RollingWindowConfig", () => {
  it("accepts default-like values", () => {
    const config: RollingWindowConfig = {
      enabled: true,
      keepSystem: true,
      keepLastTurns: 2,
      outputBufferTokens: 4000,
    };
    expect(config.keepSystem).toBe(true);
    expect(config.keepLastTurns).toBe(2);
    expect(config.outputBufferTokens).toBe(4000);
  });
});

describe("ScoringWeights", () => {
  it("accepts default-like values", () => {
    const weights: ScoringWeights = {
      recency: 0.20,
      semanticSimilarity: 0.20,
      toinImportance: 0.25,
      errorIndicator: 0.15,
      forwardReference: 0.15,
      tokenDensity: 0.05,
    };
    const total =
      weights.recency! +
      weights.semanticSimilarity! +
      weights.toinImportance! +
      weights.errorIndicator! +
      weights.forwardReference! +
      weights.tokenDensity!;
    expect(total).toBe(1.0);
  });
});

describe("IntelligentContextConfig", () => {
  it("accepts default-like values", () => {
    const config: IntelligentContextConfig = {
      enabled: true,
      keepSystem: true,
      keepLastTurns: 2,
      outputBufferTokens: 4000,
      useImportanceScoring: true,
      toinIntegration: true,
      toinConfidenceThreshold: 0.3,
      compressThreshold: 0.10,
      summarizationEnabled: false,
    };
    expect(config.enabled).toBe(true);
    expect(config.compressThreshold).toBe(0.10);
    expect(config.summarizationEnabled).toBe(false);
  });
});

describe("RelevanceScorerConfig", () => {
  it("default tier is hybrid", () => {
    const config: RelevanceScorerConfig = { tier: "hybrid" };
    expect(config.tier).toBe("hybrid");
  });

  it("accepts all tier values", () => {
    const tiers: RelevanceScorerConfig["tier"][] = ["bm25", "embedding", "hybrid"];
    expect(tiers).toHaveLength(3);
  });
});

describe("AnchorConfig", () => {
  it("accepts default-like values", () => {
    const config: AnchorConfig = {
      anchorBudgetPct: 0.25,
      minAnchorSlots: 3,
      maxAnchorSlots: 12,
      defaultFrontWeight: 0.5,
      defaultBackWeight: 0.4,
      defaultMiddleWeight: 0.1,
      useInformationDensity: true,
      dedupIdenticalItems: true,
    };
    const total =
      config.defaultFrontWeight! +
      config.defaultBackWeight! +
      config.defaultMiddleWeight!;
    expect(total).toBe(1.0);
  });
});

describe("SmartCrusherConfig", () => {
  it("accepts default-like values", () => {
    const config: SmartCrusherConfig = {
      enabled: true,
      minItemsToAnalyze: 5,
      minTokensToCrush: 200,
      varianceThreshold: 2.0,
      uniquenessThreshold: 0.1,
      similarityThreshold: 0.8,
      maxItemsAfterCrush: 15,
      preserveChangePoints: true,
      useFeedbackHints: true,
      dedupIdenticalItems: true,
    };
    expect(config.enabled).toBe(true);
    expect(config.minItemsToAnalyze).toBe(5);
  });

  it("supports nested relevance config", () => {
    const config: SmartCrusherConfig = {
      relevance: { tier: "hybrid", relevanceThreshold: 0.25 },
      anchor: { anchorBudgetPct: 0.25 },
    };
    expect(config.relevance!.tier).toBe("hybrid");
    expect(config.anchor!.anchorBudgetPct).toBe(0.25);
  });
});

describe("CacheOptimizerConfig", () => {
  it("accepts default-like values", () => {
    const config: CacheOptimizerConfig = {
      enabled: true,
      autoDetectProvider: true,
      minCacheableTokens: 1024,
      enableSemanticCache: false,
    };
    expect(config.enabled).toBe(true);
    expect(config.minCacheableTokens).toBe(1024);
  });
});

describe("CCRConfig", () => {
  it("accepts default-like values", () => {
    const config: CCRConfig = {
      enabled: true,
      storeMaxEntries: 1000,
      storeTtlSeconds: 300,
      injectRetrievalMarker: true,
      feedbackEnabled: true,
      minItemsToCache: 20,
      injectTool: true,
      injectSystemInstructions: false,
    };
    expect(config.enabled).toBe(true);
    expect(config.storeMaxEntries).toBe(1000);
    expect(config.injectTool).toBe(true);
  });
});

describe("PrefixFreezeConfig", () => {
  it("accepts default-like values", () => {
    const config: PrefixFreezeConfig = {
      enabled: true,
      minCachedTokens: 1024,
      sessionTtlSeconds: 600,
      forceCompressThreshold: 0.5,
    };
    expect(config.enabled).toBe(true);
    expect(config.sessionTtlSeconds).toBe(600);
  });
});

describe("ReadLifecycleConfig", () => {
  it("accepts default-like values", () => {
    const config: ReadLifecycleConfig = {
      enabled: true,
      compressStale: true,
      compressSuperseded: false,
      minSizeBytes: 512,
    };
    expect(config.enabled).toBe(true);
    expect(config.compressSuperseded).toBe(false);
  });
});

describe("CompressionProfile", () => {
  it("accepts default-like values", () => {
    const profile: CompressionProfile = {
      bias: 1.0,
      minK: 3,
      maxK: null,
    };
    expect(profile.bias).toBe(1.0);
    expect(profile.maxK).toBeNull();
  });
});

describe("HeadroomConfig", () => {
  it("accepts full config with all nested objects", () => {
    const config: HeadroomConfig = {
      defaultMode: "audit",
      modelContextLimits: { "gpt-4o": 128000 },
      smartCrusher: { enabled: true },
      cacheAligner: { enabled: false },
      rollingWindow: { keepLastTurns: 2 },
      cacheOptimizer: { enabled: true },
      ccr: { enabled: true },
      prefixFreeze: { enabled: true },
      contentRouterEnabled: true,
      intelligentContext: { enabled: true },
      generateDiffArtifact: false,
    };
    expect(config.defaultMode).toBe("audit");
    expect(config.smartCrusher!.enabled).toBe(true);
    expect(config.modelContextLimits!["gpt-4o"]).toBe(128000);
  });

  it("all fields are optional", () => {
    const config: HeadroomConfig = {};
    expect(config.defaultMode).toBeUndefined();
  });
});
