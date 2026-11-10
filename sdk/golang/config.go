package headroom

// HeadroomMode mirrors the Python/TS HeadroomMode enum.
type HeadroomMode string

const (
	ModeAudit    HeadroomMode = "audit"
	ModeOptimize HeadroomMode = "optimize"
	ModeSimulate HeadroomMode = "simulate"
)

type RelevanceTier string

const (
	TierBM25      RelevanceTier = "bm25"
	TierEmbedding RelevanceTier = "embedding"
	TierHybrid    RelevanceTier = "hybrid"
)

// All sub-config structs use snake_case JSON tags so they go on the wire
// directly (no camelCase ↔ snake_case conversion needed in Go).
// `omitempty` everywhere so the proxy uses defaults for omitted fields.

type ToolCrusherConfig struct {
	Enabled          *bool                     `json:"enabled,omitempty"`
	MinTokensToCrush *int                      `json:"min_tokens_to_crush,omitempty"`
	MaxArrayItems    *int                      `json:"max_array_items,omitempty"`
	MaxStringLength  *int                      `json:"max_string_length,omitempty"`
	MaxDepth         *int                      `json:"max_depth,omitempty"`
	PreserveKeys     []string                  `json:"preserve_keys,omitempty"`
	ToolProfiles     map[string]map[string]any `json:"tool_profiles,omitempty"`
}

type CacheAlignerConfig struct {
	Enabled              *bool    `json:"enabled,omitempty"`
	UseDynamicDetector   *bool    `json:"use_dynamic_detector,omitempty"`
	DetectionTiers       []string `json:"detection_tiers,omitempty"`
	ExtraDynamicLabels   []string `json:"extra_dynamic_labels,omitempty"`
	EntropyThreshold     *float64 `json:"entropy_threshold,omitempty"`
	DatePatterns         []string `json:"date_patterns,omitempty"`
	NormalizeWhitespace  *bool    `json:"normalize_whitespace,omitempty"`
	CollapseBlankLines   *bool    `json:"collapse_blank_lines,omitempty"`
	DynamicTailSeparator *string  `json:"dynamic_tail_separator,omitempty"`
}

type RollingWindowConfig struct {
	Enabled            *bool `json:"enabled,omitempty"`
	KeepSystem         *bool `json:"keep_system,omitempty"`
	KeepLastTurns      *int  `json:"keep_last_turns,omitempty"`
	OutputBufferTokens *int  `json:"output_buffer_tokens,omitempty"`
}

type ScoringWeights struct {
	Recency            *float64 `json:"recency,omitempty"`
	SemanticSimilarity *float64 `json:"semantic_similarity,omitempty"`
	TOINImportance     *float64 `json:"toin_importance,omitempty"`
	ErrorIndicator     *float64 `json:"error_indicator,omitempty"`
	ForwardReference   *float64 `json:"forward_reference,omitempty"`
	TokenDensity       *float64 `json:"token_density,omitempty"`
}

type IntelligentContextConfig struct {
	Enabled                 *bool           `json:"enabled,omitempty"`
	KeepSystem              *bool           `json:"keep_system,omitempty"`
	KeepLastTurns           *int            `json:"keep_last_turns,omitempty"`
	OutputBufferTokens      *int            `json:"output_buffer_tokens,omitempty"`
	UseImportanceScoring    *bool           `json:"use_importance_scoring,omitempty"`
	ScoringWeights          *ScoringWeights `json:"scoring_weights,omitempty"`
	RecencyDecayRate        *float64        `json:"recency_decay_rate,omitempty"`
	TOINIntegration         *bool           `json:"toin_integration,omitempty"`
	TOINConfidenceThreshold *float64        `json:"toin_confidence_threshold,omitempty"`
	CompressThreshold       *float64        `json:"compress_threshold,omitempty"`
	SummarizationEnabled    *bool           `json:"summarization_enabled,omitempty"`
	SummarizationModel      *string         `json:"summarization_model,omitempty"`
	SummaryMaxTokens        *int            `json:"summary_max_tokens,omitempty"`
	SummarizeThreshold      *float64        `json:"summarize_threshold,omitempty"`
}

type RelevanceScorerConfig struct {
	Tier               RelevanceTier `json:"tier,omitempty"`
	BM25K1             *float64      `json:"bm25_k1,omitempty"`
	BM25B              *float64      `json:"bm25_b,omitempty"`
	EmbeddingModel     *string       `json:"embedding_model,omitempty"`
	HybridAlpha        *float64      `json:"hybrid_alpha,omitempty"`
	AdaptiveAlpha      *bool         `json:"adaptive_alpha,omitempty"`
	RelevanceThreshold *float64      `json:"relevance_threshold,omitempty"`
}

type AnchorConfig struct {
	AnchorBudgetPct       *float64 `json:"anchor_budget_pct,omitempty"`
	MinAnchorSlots        *int     `json:"min_anchor_slots,omitempty"`
	MaxAnchorSlots        *int     `json:"max_anchor_slots,omitempty"`
	DefaultFrontWeight    *float64 `json:"default_front_weight,omitempty"`
	DefaultBackWeight     *float64 `json:"default_back_weight,omitempty"`
	DefaultMiddleWeight   *float64 `json:"default_middle_weight,omitempty"`
	UseInformationDensity *bool    `json:"use_information_density,omitempty"`
	DedupIdenticalItems   *bool    `json:"dedup_identical_items,omitempty"`
}

type SmartCrusherConfig struct {
	Enabled                 *bool                  `json:"enabled,omitempty"`
	MinItemsToAnalyze       *int                   `json:"min_items_to_analyze,omitempty"`
	MinTokensToCrush        *int                   `json:"min_tokens_to_crush,omitempty"`
	VarianceThreshold       *float64               `json:"variance_threshold,omitempty"`
	UniquenessThreshold     *float64               `json:"uniqueness_threshold,omitempty"`
	SimilarityThreshold     *float64               `json:"similarity_threshold,omitempty"`
	MaxItemsAfterCrush      *int                   `json:"max_items_after_crush,omitempty"`
	PreserveChangePoints    *bool                  `json:"preserve_change_points,omitempty"`
	UseFeedbackHints        *bool                  `json:"use_feedback_hints,omitempty"`
	TOINConfidenceThreshold *float64               `json:"toin_confidence_threshold,omitempty"`
	Relevance               *RelevanceScorerConfig `json:"relevance,omitempty"`
	Anchor                  *AnchorConfig          `json:"anchor,omitempty"`
	DedupIdenticalItems     *bool                  `json:"dedup_identical_items,omitempty"`
	FirstFraction           *float64               `json:"first_fraction,omitempty"`
	LastFraction            *float64               `json:"last_fraction,omitempty"`
}

type CacheOptimizerConfig struct {
	Enabled                 *bool    `json:"enabled,omitempty"`
	AutoDetectProvider      *bool    `json:"auto_detect_provider,omitempty"`
	MinCacheableTokens      *int     `json:"min_cacheable_tokens,omitempty"`
	EnableSemanticCache     *bool    `json:"enable_semantic_cache,omitempty"`
	SemanticCacheSimilarity *float64 `json:"semantic_cache_similarity,omitempty"`
	SemanticCacheMaxEntries *int     `json:"semantic_cache_max_entries,omitempty"`
	SemanticCacheTtlSeconds *int     `json:"semantic_cache_ttl_seconds,omitempty"`
}

type CCRConfig struct {
	Enabled                  *bool   `json:"enabled,omitempty"`
	StoreMaxEntries          *int    `json:"store_max_entries,omitempty"`
	StoreTTLSeconds          *int    `json:"store_ttl_seconds,omitempty"`
	InjectRetrievalMarker    *bool   `json:"inject_retrieval_marker,omitempty"`
	FeedbackEnabled          *bool   `json:"feedback_enabled,omitempty"`
	MinItemsToCache          *int    `json:"min_items_to_cache,omitempty"`
	InjectTool               *bool   `json:"inject_tool,omitempty"`
	InjectSystemInstructions *bool   `json:"inject_system_instructions,omitempty"`
	MarkerTemplate           *string `json:"marker_template,omitempty"`
}

type PrefixFreezeConfig struct {
	Enabled                *bool `json:"enabled,omitempty"`
	MinCachedTokens        *int  `json:"min_cached_tokens,omitempty"`
	SessionTTLSeconds      *int  `json:"session_ttl_seconds,omitempty"`
	ForceCompressThreshold *int  `json:"force_compress_threshold,omitempty"`
}

type ReadLifecycleConfig struct {
	Enabled            *bool `json:"enabled,omitempty"`
	CompressStale      *bool `json:"compress_stale,omitempty"`
	CompressSuperseded *bool `json:"compress_superseded,omitempty"`
	MinSizeBytes       *int  `json:"min_size_bytes,omitempty"`
}

type Config struct {
	StoreURL             *string                   `json:"store_url,omitempty"`
	DefaultMode          HeadroomMode              `json:"default_mode,omitempty"`
	ModelContextLimits   map[string]int            `json:"model_context_limits,omitempty"`
	ToolCrusher          *ToolCrusherConfig        `json:"tool_crusher,omitempty"`
	SmartCrusher         *SmartCrusherConfig       `json:"smart_crusher,omitempty"`
	CacheAligner         *CacheAlignerConfig       `json:"cache_aligner,omitempty"`
	RollingWindow        *RollingWindowConfig      `json:"rolling_window,omitempty"`
	CacheOptimizer       *CacheOptimizerConfig     `json:"cache_optimizer,omitempty"`
	CCR                  *CCRConfig                `json:"ccr,omitempty"`
	PrefixFreeze         *PrefixFreezeConfig       `json:"prefix_freeze,omitempty"`
	ContentRouterEnabled *bool                     `json:"content_router_enabled,omitempty"`
	IntelligentContext   *IntelligentContextConfig `json:"intelligent_context,omitempty"`
	GenerateDiffArtifact *bool                     `json:"generate_diff_artifact,omitempty"`
}

// Helpers for `*T` fields when callers don't want to manage pointers.
func Bool(v bool) *bool        { return &v }
func Int(v int) *int           { return &v }
func Float(v float64) *float64 { return &v }
func String(v string) *string  { return &v }
