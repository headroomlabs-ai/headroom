/**
 * Configuration types matching Python headroom.config dataclasses.
 * All fields are optional — the proxy uses defaults for omitted values.
 */

// --- Enums ---

export type HeadroomMode = "audit" | "optimize" | "simulate";

/** Pipeline mode accepted by POST /v1/compress. Unset = default marker-free
 *  pipeline. Any other value is rejected (400) by the proxy. */
export type CompressMode = "ccr" | "lossy_inline" | "lossless_then_lossy";

/** Config accepted by the POST /v1/compress endpoint. camelCase keys are
 *  snake_cased on the wire (e.g. targetRatio -> target_ratio). Every field is
 *  optional; only these keys are honored by the endpoint. */
export interface CompressRequestConfig {
  /** Compression pipeline mode (unset = default marker-free pipeline). */
  mode?: CompressMode;
  /** Target keep-ratio (0–1) for lossy compression. -> target_ratio */
  targetRatio?: number;
  /** Also compress user-role messages (default false). -> compress_user_messages */
  compressUserMessages?: boolean;
  /** Number of most-recent messages to protect from compression. -> protect_recent */
  protectRecent?: number;
  /** Protect analysis/reasoning context from compression. -> protect_analysis_context */
  protectAnalysisContext?: boolean;
  /** Freeze the first N messages (prefix) from compression. -> frozen_message_count */
  frozenMessageCount?: number;
  /** Session id for session-aware compression. -> session_id */
  sessionId?: string;
}

export type RelevanceTier = "bm25" | "embedding" | "hybrid";

export type ContentType =
  | "json"
  | "code"
  | "logs"
  | "text"
  | "html"
  | "diff"
  | "search"
  | "unknown";

export type BlockKind =
  | "system"
  | "user"
  | "assistant"
  | "tool_call"
  | "tool_result"
  | "rag"
  | "unknown";

// --- Config interfaces ---

export interface ToolCrusherConfig {
  enabled?: boolean;
  minTokensToCrush?: number;
  maxArrayItems?: number;
  maxStringLength?: number;
  maxDepth?: number;
  preserveKeys?: string[];
  toolProfiles?: Record<string, Record<string, any>>;
}

export interface CacheAlignerConfig {
  enabled?: boolean;
  useDynamicDetector?: boolean;
  detectionTiers?: ("regex" | "ner" | "semantic")[];
  extraDynamicLabels?: string[];
  entropyThreshold?: number;
  datePatterns?: string[];
  normalizeWhitespace?: boolean;
  collapseBlankLines?: boolean;
  dynamicTailSeparator?: string;
}

export interface RollingWindowConfig {
  enabled?: boolean;
  keepSystem?: boolean;
  keepLastTurns?: number;
  outputBufferTokens?: number;
}

export interface ScoringWeights {
  recency?: number;
  semanticSimilarity?: number;
  toinImportance?: number;
  errorIndicator?: number;
  forwardReference?: number;
  tokenDensity?: number;
}

export interface IntelligentContextConfig {
  enabled?: boolean;
  keepSystem?: boolean;
  keepLastTurns?: number;
  outputBufferTokens?: number;
  useImportanceScoring?: boolean;
  scoringWeights?: ScoringWeights;
  recencyDecayRate?: number;
  toinIntegration?: boolean;
  toinConfidenceThreshold?: number;
  compressThreshold?: number;
  summarizationEnabled?: boolean;
  summarizationModel?: string | null;
  summaryMaxTokens?: number;
  summarizeThreshold?: number;
}

export interface RelevanceScorerConfig {
  tier?: RelevanceTier;
  bm25K1?: number;
  bm25B?: number;
  embeddingModel?: string;
  hybridAlpha?: number;
  adaptiveAlpha?: boolean;
  relevanceThreshold?: number;
}

export interface AnchorConfig {
  anchorBudgetPct?: number;
  minAnchorSlots?: number;
  maxAnchorSlots?: number;
  defaultFrontWeight?: number;
  defaultBackWeight?: number;
  defaultMiddleWeight?: number;
  useInformationDensity?: boolean;
  dedupIdenticalItems?: boolean;
}

export interface SmartCrusherConfig {
  enabled?: boolean;
  minItemsToAnalyze?: number;
  minTokensToCrush?: number;
  varianceThreshold?: number;
  uniquenessThreshold?: number;
  similarityThreshold?: number;
  maxItemsAfterCrush?: number;
  preserveChangePoints?: boolean;
  useFeedbackHints?: boolean;
  toinConfidenceThreshold?: number;
  relevance?: RelevanceScorerConfig;
  anchor?: AnchorConfig;
  dedupIdenticalItems?: boolean;
  firstFraction?: number;
  lastFraction?: number;
}

export interface CacheOptimizerConfig {
  enabled?: boolean;
  autoDetectProvider?: boolean;
  minCacheableTokens?: number;
  enableSemanticCache?: boolean;
  semanticCacheSimilarity?: number;
  semanticCacheMaxEntries?: number;
  semanticCacheTtlSeconds?: number;
}

export interface CCRConfig {
  enabled?: boolean;
  storeMaxEntries?: number;
  storeTtlSeconds?: number;
  injectRetrievalMarker?: boolean;
  feedbackEnabled?: boolean;
  minItemsToCache?: number;
  injectTool?: boolean;
  injectSystemInstructions?: boolean;
  markerTemplate?: string;
}

export interface PrefixFreezeConfig {
  enabled?: boolean;
  minCachedTokens?: number;
  sessionTtlSeconds?: number;
  forceCompressThreshold?: number;
}

export interface ReadLifecycleConfig {
  enabled?: boolean;
  compressStale?: boolean;
  compressSuperseded?: boolean;
  minSizeBytes?: number;
}

export interface CompressionProfile {
  bias?: number;
  minK?: number;
  maxK?: number | null;
}

export interface HeadroomConfig {
  storeUrl?: string;
  defaultMode?: HeadroomMode;
  modelContextLimits?: Record<string, number>;
  toolCrusher?: ToolCrusherConfig;
  smartCrusher?: SmartCrusherConfig;
  cacheAligner?: CacheAlignerConfig;
  rollingWindow?: RollingWindowConfig;
  cacheOptimizer?: CacheOptimizerConfig;
  ccr?: CCRConfig;
  prefixFreeze?: PrefixFreezeConfig;
  contentRouterEnabled?: boolean;
  intelligentContext?: IntelligentContextConfig;
  generateDiffArtifact?: boolean;
}
