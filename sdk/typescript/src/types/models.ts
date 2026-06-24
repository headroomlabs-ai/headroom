/**
 * Data models matching Python headroom.config result/metrics dataclasses.
 */

import type { BlockKind } from "./config.js";

// --- Waste Signals ---

export interface WasteSignals {
  jsonBloatTokens: number;
  htmlNoiseTokens: number;
  base64Tokens: number;
  whitespaceTokens: number;
  dynamicDateTokens: number;
  repetitionTokens: number;
  total: number;
}

// --- Cache Prefix Metrics ---

export interface CachePrefixMetrics {
  stablePrefixBytes: number;
  stablePrefixTokensEst: number;
  stablePrefixHash: string;
  prefixChanged: boolean;
  previousHash?: string | null;
}

// --- Transform Diff ---

export interface TransformDiff {
  transformName: string;
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  itemsRemoved: number;
  itemsKept: number;
  details: string;
  durationMs: number;
}

// --- Diff Artifact ---

export interface DiffArtifact {
  requestId: string;
  originalTokens: number;
  optimizedTokens: number;
  totalTokensSaved: number;
  transforms: TransformDiff[];
}

// --- Simulation Result ---

export interface SimulationResult {
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  transforms: string[];
  estimatedSavings: string;
  messagesOptimized: any[];
  blockBreakdown: Record<string, number>;
  wasteSignals: Record<string, number>;
  stablePrefixHash: string;
  cacheAlignmentScore: number;
}

// --- Request Metrics ---

export interface RequestMetrics {
  requestId: string;
  timestamp: string;
  model: string;
  stream: boolean;
  mode: string;
  tokensInputBefore: number;
  tokensInputAfter: number;
  tokensOutput?: number | null;
  blockBreakdown: Record<string, number>;
  wasteSignals: Record<string, number>;
  stablePrefixHash: string;
  cacheAlignmentScore: number;
  cachedTokens?: number | null;
  cacheOptimizerUsed?: string | null;
  cacheOptimizerStrategy?: string | null;
  cacheableTokens: number;
  breakpointsInserted: number;
  estimatedCacheHit: boolean;
  estimatedSavingsPercent: number;
  semanticCacheHit: boolean;
  transformsApplied: string[];
  toolUnitsDropped: number;
  turnsDropped: number;
  messagesHash: string;
  error?: string | null;
}

// --- Block ---

export interface Block {
  kind: BlockKind;
  text: string;
  tokensEst: number;
  contentHash: string;
  sourceIndex: number;
  flags: Record<string, any>;
}

// --- Session Stats ---

export interface SessionStats {
  totalRequests: number;
  totalTokensBefore: number;
  totalTokensAfter: number;
  totalTokensSaved: number;
  averageCompressionRatio: number;
  cacheHits: number;
  byMode: Record<string, { requests: number; tokensSaved: number }>;
}

// --- Validation Result ---

export interface ValidationResult {
  valid: boolean;
  provider: string;
  errors: string[];
  warnings: string[];
  config: Record<string, any>;
}

// --- Summary ---

export interface MetricsSummary {
  totalRequests: number;
  totalTokensBefore: number;
  totalTokensAfter: number;
  totalTokensSaved: number;
  averageCompressionRatio: number;
  models: Record<string, number>;
  modes: Record<string, number>;
  errorCount: number;
}

// --- Health ---

export interface HealthStatus {
  status: "healthy" | "unhealthy";
  version: string;
  config: {
    optimize: boolean;
    cache: boolean;
    rateLimit: boolean;
  };
}

// --- Proxy Stats ---

export interface ProxyStats {
  requests: {
    total: number;
    cached: number;
    rateLimited: number;
    failed: number;
    byProvider: Record<string, number>;
    byModel: Record<string, number>;
  };
  tokens: {
    input: number;
    output: number;
    saved: number;
    cliTokensAvoided: number;
    totalBeforeCompression: number;
    savingsPercent: number;
  };
  latency: { averageMs: number; minMs: number; maxMs: number };
  overhead: { averageMs: number; minMs: number; maxMs: number };
  pipelineTiming: Record<
    string,
    { averageMs: number; maxMs: number; count: number }
  >;
  wasteSignals: Record<string, number>;
  compression: {
    ccrEntries: number;
    ccrMaxEntries: number;
    originalTokensCached: number;
    compressedTokensCached: number;
    ccrRetrievals: number;
  };
  cost: Record<string, any>;
  feedbackLoop: {
    toolsTracked: number;
    totalCompressions: number;
    totalRetrievals: number;
    globalRetrievalRate: number;
  };
}

// --- Memory Usage ---

export interface MemoryUsage {
  processMemory: { rss: number; vms: number; percent: number };
  components: Record<string, { memoryMb: number; budgetMb: number }>;
  totalTrackedMb: number;
  targetBudgetMb: number;
}

// --- CCR Types ---

export interface RetrieveResult {
  hash: string;
  originalContent: string;
  originalTokens: number;
  originalItemCount: number;
  compressedItemCount: number;
  toolName: string;
  retrievalCount: number;
}

export interface RetrieveSearchResult {
  hash: string;
  query: string;
  results: any[];
  count: number;
}

export interface CCRStats {
  store: {
    entries: number;
    maxEntries: number;
    originalTokensCached: number;
    compressedTokensCached: number;
    retrievals: number;
  };
  recentRetrievals: Array<{
    hash: string;
    query: string | null;
    itemsRetrieved: number;
    totalItems: number;
    toolName: string;
    retrievalType: string;
  }>;
}

// --- Telemetry ---

export interface TelemetryStats {
  enabled: boolean;
  totalCompressions: number;
  totalRetrievals: number;
  globalRetrievalRate: number;
  toolSignaturesTracked: number;
  avgCompressionRatio: number;
  avgTokenReduction: number;
}

export interface ToolHints {
  toolName: string;
  hints: {
    maxItems: number;
    minItems: number;
    suggestedItems: any[];
    aggressiveness: number;
    skipCompression: boolean;
    preserveFields: string[];
    reason: string;
  };
  pattern: {
    totalCompressions: number;
    totalRetrievals: number;
    retrievalRate: number;
    fullRetrievalRate: number;
    searchRate: number;
    commonQueries: string[];
    queriedFields: string[];
  };
}

export interface TOINStats {
  enabled: boolean;
  patternsTracked: number;
  totalCompressions: number;
  totalRetrievals: number;
  globalRetrievalRate: number;
  patternsWithRecommendations: number;
}

export interface TOINPattern {
  hash: string;
  compressions: number;
  retrievals: number;
  retrievalRate: string;
  confidence: number;
  skipRecommended: boolean;
  optimalMaxItems: number;
}

// --- Metrics Query ---

export interface MetricsQuery {
  startTime?: Date;
  endTime?: Date;
  model?: string;
  mode?: string;
  limit?: number;
}

export interface SummaryQuery {
  startTime?: Date;
  endTime?: Date;
}

export interface StatsHistoryQuery {
  format?: "json" | "csv";
  series?: "history" | "hourly" | "daily" | "weekly" | "monthly";
}
