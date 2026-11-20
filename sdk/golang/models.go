package headroom

// Data models matching headroom server responses. JSON tags follow the
// proxy's snake_case wire format directly.

type WasteSignals struct {
	JSONBloatTokens   int `json:"json_bloat_tokens"`
	HTMLNoiseTokens   int `json:"html_noise_tokens"`
	Base64Tokens      int `json:"base64_tokens"`
	WhitespaceTokens  int `json:"whitespace_tokens"`
	DynamicDateTokens int `json:"dynamic_date_tokens"`
	RepetitionTokens  int `json:"repetition_tokens"`
	Total             int `json:"total"`
}

type CachePrefixMetrics struct {
	StablePrefixBytes     int    `json:"stable_prefix_bytes"`
	StablePrefixTokensEst int    `json:"stable_prefix_tokens_est"`
	StablePrefixHash      string `json:"stable_prefix_hash"`
	PrefixChanged         bool   `json:"prefix_changed"`
	PreviousHash          string `json:"previous_hash,omitempty"`
}

type TransformDiff struct {
	TransformName string  `json:"transform_name"`
	TokensBefore  int     `json:"tokens_before"`
	TokensAfter   int     `json:"tokens_after"`
	TokensSaved   int     `json:"tokens_saved"`
	ItemsRemoved  int     `json:"items_removed"`
	ItemsKept     int     `json:"items_kept"`
	Details       string  `json:"details"`
	DurationMs    float64 `json:"duration_ms"`
}

type DiffArtifact struct {
	RequestID        string          `json:"request_id"`
	OriginalTokens   int             `json:"original_tokens"`
	OptimizedTokens  int             `json:"optimized_tokens"`
	TotalTokensSaved int             `json:"total_tokens_saved"`
	Transforms       []TransformDiff `json:"transforms"`
}

type SimulationResult struct {
	TokensBefore        int            `json:"tokens_before"`
	TokensAfter         int            `json:"tokens_after"`
	TokensSaved         int            `json:"tokens_saved"`
	Transforms          []string       `json:"transforms"`
	EstimatedSavings    string         `json:"estimated_savings"`
	MessagesOptimized   []any          `json:"messages_optimized"`
	BlockBreakdown      map[string]int `json:"block_breakdown"`
	WasteSignals        map[string]int `json:"waste_signals"`
	StablePrefixHash    string         `json:"stable_prefix_hash"`
	CacheAlignmentScore float64        `json:"cache_alignment_score"`
}

type RequestMetrics struct {
	RequestID              string         `json:"request_id"`
	Timestamp              string         `json:"timestamp"`
	Model                  string         `json:"model"`
	Stream                 bool           `json:"stream"`
	Mode                   string         `json:"mode"`
	TokensInputBefore      int            `json:"tokens_input_before"`
	TokensInputAfter       int            `json:"tokens_input_after"`
	TokensOutput           *int           `json:"tokens_output,omitempty"`
	BlockBreakdown         map[string]int `json:"block_breakdown"`
	WasteSignals           map[string]int `json:"waste_signals"`
	StablePrefixHash       string         `json:"stable_prefix_hash"`
	CacheAlignmentScore    float64        `json:"cache_alignment_score"`
	CachedTokens           *int           `json:"cached_tokens,omitempty"`
	CacheOptimizerUsed     *string        `json:"cache_optimizer_used,omitempty"`
	CacheOptimizerStrategy *string        `json:"cache_optimizer_strategy,omitempty"`
	CacheableTokens        int            `json:"cacheable_tokens"`
	BreakpointsInserted    int            `json:"breakpoints_inserted"`
	EstimatedCacheHit      bool           `json:"estimated_cache_hit"`
	EstimatedSavingsPct    float64        `json:"estimated_savings_percent"`
	SemanticCacheHit       bool           `json:"semantic_cache_hit"`
	TransformsApplied      []string       `json:"transforms_applied"`
	ToolUnitsDropped       int            `json:"tool_units_dropped"`
	TurnsDropped           int            `json:"turns_dropped"`
	MessagesHash           string         `json:"messages_hash"`
	Error                  *string        `json:"error,omitempty"`
}

type SessionStats struct {
	TotalRequests           int                       `json:"total_requests"`
	TotalTokensBefore       int                       `json:"total_tokens_before"`
	TotalTokensAfter        int                       `json:"total_tokens_after"`
	TotalTokensSaved        int                       `json:"total_tokens_saved"`
	AverageCompressionRatio float64                   `json:"average_compression_ratio"`
	CacheHits               int                       `json:"cache_hits"`
	ByMode                  map[string]ModeStatsEntry `json:"by_mode"`
}

type ModeStatsEntry struct {
	Requests    int `json:"requests"`
	TokensSaved int `json:"tokens_saved"`
}

type ValidationResult struct {
	Valid    bool           `json:"valid"`
	Provider string         `json:"provider"`
	Errors   []string       `json:"errors"`
	Warnings []string       `json:"warnings"`
	Config   map[string]any `json:"config"`
}

type MetricsSummary struct {
	TotalRequests           int            `json:"total_requests"`
	TotalTokensBefore       int            `json:"total_tokens_before"`
	TotalTokensAfter        int            `json:"total_tokens_after"`
	TotalTokensSaved        int            `json:"total_tokens_saved"`
	AverageCompressionRatio float64        `json:"average_compression_ratio"`
	Models                  map[string]int `json:"models"`
	Modes                   map[string]int `json:"modes"`
	ErrorCount              int            `json:"error_count"`
}

// HealthStatus models /health. The proxy returns a richer payload than the
// TS SDK's interface; we keep the common fields and surface the rest as raw.
type HealthStatus struct {
	Status  string         `json:"status"`
	Version string         `json:"version"`
	Ready   bool           `json:"ready"`
	Service string         `json:"service,omitempty"`
	Config  HealthConfig   `json:"config"`
	Checks  map[string]any `json:"checks,omitempty"`
}

type HealthConfig struct {
	Backend   string `json:"backend,omitempty"`
	Optimize  bool   `json:"optimize"`
	Cache     bool   `json:"cache"`
	RateLimit bool   `json:"rate_limit"`
}

type ProxyStats struct {
	Requests       ProxyStatsRequests             `json:"requests"`
	Tokens         ProxyStatsTokens               `json:"tokens"`
	Latency        LatencyStats                   `json:"latency"`
	Overhead       LatencyStats                   `json:"overhead"`
	PipelineTiming map[string]PipelineTimingEntry `json:"pipeline_timing"`
	WasteSignals   map[string]int                 `json:"waste_signals"`
	Compression    ProxyStatsCompression          `json:"compression"`
	Cost           map[string]any                 `json:"cost"`
	FeedbackLoop   ProxyStatsFeedbackLoop         `json:"feedback_loop"`
	RecentRequests []map[string]any               `json:"recent_requests,omitempty"`
}

type ProxyStatsRequests struct {
	Total       int            `json:"total"`
	Cached      int            `json:"cached"`
	RateLimited int            `json:"rate_limited"`
	Failed      int            `json:"failed"`
	ByProvider  map[string]int `json:"by_provider"`
	ByModel     map[string]int `json:"by_model"`
}

type ProxyStatsTokens struct {
	Input                  int     `json:"input"`
	Output                 int     `json:"output"`
	Saved                  int     `json:"saved"`
	CLITokensAvoided       int     `json:"cli_tokens_avoided"`
	TotalBeforeCompression int     `json:"total_before_compression"`
	SavingsPercent         float64 `json:"savings_percent"`
}

type LatencyStats struct {
	AverageMs float64 `json:"average_ms"`
	MinMs     float64 `json:"min_ms"`
	MaxMs     float64 `json:"max_ms"`
}

type PipelineTimingEntry struct {
	AverageMs float64 `json:"average_ms"`
	MaxMs     float64 `json:"max_ms"`
	Count     int     `json:"count"`
}

type ProxyStatsCompression struct {
	CCREntries             int `json:"ccr_entries"`
	CCRMaxEntries          int `json:"ccr_max_entries"`
	OriginalTokensCached   int `json:"original_tokens_cached"`
	CompressedTokensCached int `json:"compressed_tokens_cached"`
	CCRRetrievals          int `json:"ccr_retrievals"`
}

type ProxyStatsFeedbackLoop struct {
	ToolsTracked        int     `json:"tools_tracked"`
	TotalCompressions   int     `json:"total_compressions"`
	TotalRetrievals     int     `json:"total_retrievals"`
	GlobalRetrievalRate float64 `json:"global_retrieval_rate"`
}

type MemoryUsage struct {
	ProcessMemory  ProcessMemory              `json:"process_memory"`
	Components     map[string]ComponentMemory `json:"components"`
	TotalTrackedMb float64                    `json:"total_tracked_mb"`
	TargetBudgetMb float64                    `json:"target_budget_mb"`
}

type ProcessMemory struct {
	RSS     int64   `json:"rss"`
	VMS     int64   `json:"vms"`
	Percent float64 `json:"percent"`
}

type ComponentMemory struct {
	MemoryMb float64 `json:"memory_mb"`
	BudgetMb float64 `json:"budget_mb"`
}

type RetrieveResult struct {
	Hash                string `json:"hash"`
	OriginalContent     string `json:"original_content"`
	OriginalTokens      int    `json:"original_tokens"`
	OriginalItemCount   int    `json:"original_item_count"`
	CompressedItemCount int    `json:"compressed_item_count"`
	ToolName            string `json:"tool_name"`
	RetrievalCount      int    `json:"retrieval_count"`
	// Search variant
	Query   string `json:"query,omitempty"`
	Results []any  `json:"results,omitempty"`
	Count   int    `json:"count,omitempty"`
}

type CCRStats struct {
	Store struct {
		Entries                int `json:"entries"`
		MaxEntries             int `json:"max_entries"`
		OriginalTokensCached   int `json:"original_tokens_cached"`
		CompressedTokensCached int `json:"compressed_tokens_cached"`
		Retrievals             int `json:"retrievals"`
	} `json:"store"`
	RecentRetrievals []map[string]any `json:"recent_retrievals"`
}

type TelemetryStats struct {
	Enabled               bool    `json:"enabled"`
	TotalCompressions     int     `json:"total_compressions"`
	TotalRetrievals       int     `json:"total_retrievals"`
	GlobalRetrievalRate   float64 `json:"global_retrieval_rate"`
	ToolSignaturesTracked int     `json:"tool_signatures_tracked"`
	AvgCompressionRatio   float64 `json:"avg_compression_ratio"`
	AvgTokenReduction     float64 `json:"avg_token_reduction"`
}

type ToolHints struct {
	ToolName string         `json:"tool_name"`
	Hints    map[string]any `json:"hints"`
	Pattern  map[string]any `json:"pattern"`
}

type TOINStats struct {
	Enabled                     bool    `json:"enabled"`
	PatternsTracked             int     `json:"patterns_tracked"`
	TotalCompressions           int     `json:"total_compressions"`
	TotalRetrievals             int     `json:"total_retrievals"`
	GlobalRetrievalRate         float64 `json:"global_retrieval_rate"`
	PatternsWithRecommendations int     `json:"patterns_with_recommendations"`
}

type TOINPattern struct {
	Hash            string  `json:"hash"`
	Compressions    int     `json:"compressions"`
	Retrievals      int     `json:"retrievals"`
	RetrievalRate   string  `json:"retrieval_rate"`
	Confidence      float64 `json:"confidence"`
	SkipRecommended bool    `json:"skip_recommended"`
	OptimalMaxItems int     `json:"optimal_max_items"`
}

// Query option structs.
type MetricsQuery struct {
	Model string
	Mode  string
	Limit int
}

type StatsHistoryQuery struct {
	Format string // "json" | "csv"
	Series string // "history" | "hourly" | "daily" | "weekly" | "monthly"
}
