package headroom

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultBaseURL = "http://localhost:8787"
	defaultTimeout = 30 * time.Second
	defaultRetries = 1
)

// Client talks to the Headroom proxy over HTTP.
type Client struct {
	baseURL        string
	apiKey         string
	providerAPIKey string
	timeout        time.Duration
	retries        int
	fallback       bool
	httpClient     *http.Client
	config         *Config
	defaultMode    HeadroomMode

	Telemetry telemetryAPI
	Feedback  feedbackAPI
	TOIN      toinAPI
}

// NewClient builds a new Headroom client.
func NewClient(opts ...Option) *Client {
	o := ClientOptions{}
	for _, fn := range opts {
		fn(&o)
	}
	return newClientFromOptions(o)
}

func newClientFromOptions(o ClientOptions) *Client {
	baseURL := o.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("HEADROOM_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	apiKey := o.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("HEADROOM_API_KEY")
	}

	timeout := o.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}
	retries := o.Retries
	if retries == 0 {
		retries = defaultRetries
	}
	fallback := true
	if o.Fallback != nil {
		fallback = *o.Fallback
	}

	httpClient := o.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: timeout}
	}

	c := &Client{
		baseURL:        baseURL,
		apiKey:         apiKey,
		providerAPIKey: o.ProviderAPIKey,
		timeout:        timeout,
		retries:        retries,
		fallback:       fallback,
		httpClient:     httpClient,
		config:         o.Config,
		defaultMode:    o.DefaultMode,
	}
	c.Telemetry = telemetryAPI{c: c}
	c.Feedback = feedbackAPI{c: c}
	c.TOIN = toinAPI{c: c}
	return c
}

// Close is a no-op (HTTP is stateless); included for parity with Python/TS.
func (c *Client) Close() {}

// =====================================================================
// Core: compress
// =====================================================================

// Compress sends OpenAI-format messages to /v1/compress.
// On transient failures it retries (up to retries+1 attempts) then either
// falls back to the original messages or returns the last error,
// depending on the fallback setting.
func (c *Client) Compress(ctx context.Context, messages []Message, opts CompressOptions) (*CompressResult, error) {
	model := opts.Model
	if model == "" {
		model = "gpt-4o"
	}

	maxAttempts := 1 + c.retries
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		res, err := c.doCompress(ctx, messages, model, opts.TokenBudget)
		if err == nil {
			return res, nil
		}
		lastErr = err
		// Auth errors and 4xx are not retryable.
		if errors.Is(err, ErrAuth) {
			return nil, err
		}
		var ce *CompressError
		if errors.As(err, &ce) && ce.StatusCode < 500 {
			return nil, err
		}
	}

	if c.fallback {
		return &CompressResult{Messages: messages, CompressionRatio: 1.0, Compressed: false}, nil
	}
	return nil, lastErr
}

// CompressRaw sends an arbitrary body to /v1/compress and returns the raw decoded JSON.
// Used by simulate() and other advanced flows.
func (c *Client) CompressRaw(ctx context.Context, body map[string]any) (map[string]any, error) {
	resp, err := c.do(ctx, http.MethodPost, "/v1/compress", body, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode compress response: %w", err)
	}
	return out, nil
}

func (c *Client) doCompress(ctx context.Context, messages []Message, model string, tokenBudget int) (*CompressResult, error) {
	body := map[string]any{
		"messages": messages,
		"model":    model,
	}
	if tokenBudget > 0 {
		body["token_budget"] = tokenBudget
	}
	if c.config != nil {
		body["config"] = c.config
	}

	resp, err := c.do(ctx, http.MethodPost, "/v1/compress", body, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var data proxyCompressResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("decode compress response: %w", err)
	}
	return &CompressResult{
		Messages:          data.Messages,
		TokensBefore:      data.TokensBefore,
		TokensAfter:       data.TokensAfter,
		TokensSaved:       data.TokensSaved,
		CompressionRatio:  data.CompressionRatio,
		TransformsApplied: data.TransformsApplied,
		CCRHashes:         data.CCRHashes,
		Compressed:        true,
	}, nil
}

// =====================================================================
// Health & Stats
// =====================================================================

func (c *Client) Health(ctx context.Context) (*HealthStatus, error) {
	var out HealthStatus
	if err := c.getJSON(ctx, "/health", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) ProxyStats(ctx context.Context) (*ProxyStats, error) {
	var out ProxyStats
	if err := c.getJSON(ctx, "/stats", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) PrometheusMetrics(ctx context.Context) (string, error) {
	resp, err := c.do(ctx, http.MethodGet, "/metrics", nil, nil)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (c *Client) StatsHistory(ctx context.Context, q *StatsHistoryQuery) (map[string]any, error) {
	path := "/stats-history"
	if q != nil {
		v := url.Values{}
		if q.Format != "" {
			v.Set("format", q.Format)
		}
		if q.Series != "" {
			v.Set("series", q.Series)
		}
		if s := v.Encode(); s != "" {
			path += "?" + s
		}
	}
	var out map[string]any
	if err := c.getJSON(ctx, path, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (c *Client) MemoryUsage(ctx context.Context) (*MemoryUsage, error) {
	var out MemoryUsage
	if err := c.getJSON(ctx, "/debug/memory", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) ClearCache(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	resp, err := c.do(ctx, http.MethodPost, "/cache/clear", nil, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// =====================================================================
// Metrics
// =====================================================================

// GetMetrics fetches /stats and filters recent_requests client-side.
func (c *Client) GetMetrics(ctx context.Context, q *MetricsQuery) ([]map[string]any, error) {
	var stats map[string]any
	if err := c.getJSON(ctx, "/stats", &stats); err != nil {
		return nil, err
	}
	rawList, _ := stats["recent_requests"].([]any)
	out := make([]map[string]any, 0, len(rawList))
	for _, r := range rawList {
		m, ok := r.(map[string]any)
		if !ok {
			continue
		}
		if q != nil {
			if q.Model != "" && m["model"] != q.Model {
				continue
			}
			if q.Mode != "" && m["mode"] != q.Mode {
				continue
			}
		}
		out = append(out, m)
		if q != nil && q.Limit > 0 && len(out) >= q.Limit {
			break
		}
	}
	return out, nil
}

func (c *Client) GetSummary(ctx context.Context) (*MetricsSummary, error) {
	var stats struct {
		Requests struct {
			Total   int            `json:"total"`
			Failed  int            `json:"failed"`
			ByModel map[string]int `json:"by_model"`
		} `json:"requests"`
		Tokens struct {
			TotalBeforeCompression int     `json:"total_before_compression"`
			Saved                  int     `json:"saved"`
			SavingsPercent         float64 `json:"savings_percent"`
		} `json:"tokens"`
	}
	if err := c.getJSON(ctx, "/stats", &stats); err != nil {
		return nil, err
	}
	var ratio float64
	if stats.Tokens.SavingsPercent != 0 {
		ratio = stats.Tokens.SavingsPercent / 100.0
	}
	return &MetricsSummary{
		TotalRequests:           stats.Requests.Total,
		TotalTokensBefore:       stats.Tokens.TotalBeforeCompression,
		TotalTokensAfter:        stats.Tokens.TotalBeforeCompression - stats.Tokens.Saved,
		TotalTokensSaved:        stats.Tokens.Saved,
		AverageCompressionRatio: ratio,
		Models:                  stats.Requests.ByModel,
		Modes:                   map[string]int{},
		ErrorCount:              stats.Requests.Failed,
	}, nil
}

func (c *Client) GetStats(ctx context.Context) (*SessionStats, error) {
	var stats struct {
		Requests struct {
			Total  int `json:"total"`
			Cached int `json:"cached"`
		} `json:"requests"`
		Tokens struct {
			TotalBeforeCompression int     `json:"total_before_compression"`
			Saved                  int     `json:"saved"`
			SavingsPercent         float64 `json:"savings_percent"`
		} `json:"tokens"`
	}
	if err := c.getJSON(ctx, "/stats", &stats); err != nil {
		return nil, err
	}
	var ratio float64
	if stats.Tokens.SavingsPercent != 0 {
		ratio = stats.Tokens.SavingsPercent / 100.0
	}
	return &SessionStats{
		TotalRequests:           stats.Requests.Total,
		TotalTokensBefore:       stats.Tokens.TotalBeforeCompression,
		TotalTokensAfter:        stats.Tokens.TotalBeforeCompression - stats.Tokens.Saved,
		TotalTokensSaved:        stats.Tokens.Saved,
		AverageCompressionRatio: ratio,
		CacheHits:               stats.Requests.Cached,
		ByMode:                  map[string]ModeStatsEntry{},
	}, nil
}

func (c *Client) ValidateSetup(ctx context.Context) (*ValidationResult, error) {
	h, err := c.Health(ctx)
	if err != nil {
		return &ValidationResult{Valid: false, Errors: []string{err.Error()}}, nil
	}
	out := &ValidationResult{
		Valid:    h.Status == "healthy",
		Provider: h.Config.Backend,
		Errors:   []string{},
		Warnings: []string{},
		Config: map[string]any{
			"optimize":   h.Config.Optimize,
			"cache":      h.Config.Cache,
			"rate_limit": h.Config.RateLimit,
			"backend":    h.Config.Backend,
		},
	}
	if !out.Valid {
		out.Errors = append(out.Errors, "Proxy unhealthy")
	}
	return out, nil
}

// =====================================================================
// CCR retrieve
// =====================================================================

// Retrieve returns the full original content stored under hash.
func (c *Client) Retrieve(ctx context.Context, hash string) (*RetrieveResult, error) {
	body := map[string]any{"hash": hash}
	resp, err := c.do(ctx, http.MethodPost, "/v1/retrieve", body, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out RetrieveResult
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) GetCCRStats(ctx context.Context) (*CCRStats, error) {
	var out CCRStats
	if err := c.getJSON(ctx, "/v1/retrieve/stats", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) HandleToolCall(ctx context.Context, toolCall any, provider string) (map[string]any, error) {
	body := map[string]any{"tool_call": toolCall}
	if provider != "" {
		body["provider"] = provider
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/retrieve/tool_call", body, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// =====================================================================
// Telemetry / Feedback / TOIN sub-namespaces
// =====================================================================

type telemetryAPI struct{ c *Client }

func (t telemetryAPI) GetStats(ctx context.Context) (*TelemetryStats, error) {
	var out TelemetryStats
	if err := t.c.getJSON(ctx, "/v1/telemetry", &out); err != nil {
		return nil, err
	}
	return &out, nil
}
func (t telemetryAPI) Export(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	if err := t.c.getJSON(ctx, "/v1/telemetry/export", &out); err != nil {
		return nil, err
	}
	return out, nil
}
func (t telemetryAPI) Import(ctx context.Context, data map[string]any) (map[string]any, error) {
	resp, err := t.c.do(ctx, http.MethodPost, "/v1/telemetry/import", data, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return out, nil
}
func (t telemetryAPI) GetTools(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	if err := t.c.getJSON(ctx, "/v1/telemetry/tools", &out); err != nil {
		return nil, err
	}
	return out, nil
}
func (t telemetryAPI) GetTool(ctx context.Context, signatureHash string) (map[string]any, error) {
	var out map[string]any
	if err := t.c.getJSON(ctx, "/v1/telemetry/tools/"+url.PathEscape(signatureHash), &out); err != nil {
		return nil, err
	}
	return out, nil
}

type feedbackAPI struct{ c *Client }

func (f feedbackAPI) GetStats(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	if err := f.c.getJSON(ctx, "/v1/feedback", &out); err != nil {
		return nil, err
	}
	return out, nil
}
func (f feedbackAPI) GetHints(ctx context.Context, toolName string) (*ToolHints, error) {
	var out ToolHints
	if err := f.c.getJSON(ctx, "/v1/feedback/"+url.PathEscape(toolName), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

type toinAPI struct{ c *Client }

func (t toinAPI) GetStats(ctx context.Context) (*TOINStats, error) {
	var out TOINStats
	if err := t.c.getJSON(ctx, "/v1/toin/stats", &out); err != nil {
		return nil, err
	}
	return &out, nil
}
func (t toinAPI) GetPatterns(ctx context.Context, limit int) ([]TOINPattern, error) {
	path := "/v1/toin/patterns"
	if limit > 0 {
		path += "?limit=" + strconv.Itoa(limit)
	}
	var out []TOINPattern
	if err := t.c.getJSON(ctx, path, &out); err != nil {
		return nil, err
	}
	return out, nil
}
func (t toinAPI) GetPattern(ctx context.Context, hashPrefix string) (map[string]any, error) {
	var out map[string]any
	if err := t.c.getJSON(ctx, "/v1/toin/pattern/"+url.PathEscape(hashPrefix), &out); err != nil {
		return nil, err
	}
	return out, nil
}

// =====================================================================
// Internal HTTP plumbing
// =====================================================================

// do performs an authed request. extraHeaders override defaults (e.g. provider
// auth headers for chat.completions / messages).
func (c *Client) do(ctx context.Context, method, path string, body any, extraHeaders map[string]string) (*http.Response, error) {
	var buf io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal body: %w", err)
		}
		buf = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, buf)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	hasAuth := false
	hasXAPIKey := false
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
		if strings.EqualFold(k, "Authorization") {
			hasAuth = true
		}
		if strings.EqualFold(k, "x-api-key") {
			hasXAPIKey = true
		}
	}
	if c.apiKey != "" && !hasAuth && !hasXAPIKey {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, wrapConnection("failed to connect to Headroom at %s: %v", c.baseURL, err)
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		var pe proxyErrorResponse
		_ = json.Unmarshal(b, &pe)
		errType := pe.Error.Type
		if errType == "" {
			errType = "unknown"
		}
		msg := pe.Error.Message
		if msg == "" {
			msg = fmt.Sprintf("HTTP %d", resp.StatusCode)
		}
		return nil, MapProxyError(resp.StatusCode, errType, msg)
	}
	return resp, nil
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	resp, err := c.do(ctx, http.MethodGet, path, nil, nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(out)
}

// rawFetch is exposed for adapters/streaming consumers.
// Returns the raw http.Response (caller closes Body).
func (c *Client) rawFetch(ctx context.Context, method, path string, body any, extraHeaders map[string]string) (*http.Response, error) {
	return c.do(ctx, method, path, body, extraHeaders)
}

// BaseURL exposes the configured proxy base URL.
func (c *Client) BaseURL() string { return c.baseURL }

// ProviderAPIKey returns the configured upstream provider key (or env fallback).
func (c *Client) ProviderAPIKey(envName string) string {
	if c.providerAPIKey != "" {
		return c.providerAPIKey
	}
	return os.Getenv(envName)
}
