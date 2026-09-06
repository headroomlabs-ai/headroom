package headroom_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	headroom "github.com/headroomlabs-ai/headroom/sdk/golang"
	"github.com/headroomlabs-ai/headroom/sdk/golang/anthropic"
	"github.com/headroomlabs-ai/headroom/sdk/golang/format"
	"github.com/headroomlabs-ai/headroom/sdk/golang/gemini"
	sdkstream "github.com/headroomlabs-ai/headroom/sdk/golang/stream"
)

// Each test exercises one Client method and asserts the request reached the
// proxy and a typed response decoded. Many endpoints return empty-but-valid
// data on a fresh proxy — that's fine; we're testing the SDK's wiring.

func TestParity_ClearCache(t *testing.T) {
	c := newTestClient(t)
	out, err := c.ClearCache(context.Background())
	if err != nil {
		t.Fatalf("clear cache: %v", err)
	}
	if out["status"] == nil {
		t.Errorf("expected status field, got %v", out)
	}
}

func TestParity_StatsHistory(t *testing.T) {
	c := newTestClient(t)
	out, err := c.StatsHistory(context.Background(), nil)
	if err != nil {
		t.Fatalf("stats history: %v", err)
	}
	if out == nil {
		t.Error("expected non-nil response")
	}
}

func TestParity_StatsHistoryWithQuery(t *testing.T) {
	c := newTestClient(t)
	out, err := c.StatsHistory(context.Background(), &headroom.StatsHistoryQuery{Series: "hourly"})
	if err != nil {
		t.Fatalf("stats history: %v", err)
	}
	if out == nil {
		t.Error("expected non-nil response")
	}
}

func TestParity_MemoryUsage(t *testing.T) {
	c := newTestClient(t)
	out, err := c.MemoryUsage(context.Background())
	if err != nil {
		t.Fatalf("memory usage: %v", err)
	}
	if out == nil {
		t.Fatal("nil response")
	}
}

func TestParity_GetMetrics(t *testing.T) {
	c := newTestClient(t)
	// Drive at least one request so /stats has recent_requests data.
	_, _ = c.Compress(context.Background(), []headroom.Message{
		{Role: "user", Content: headroom.TextContent("test")},
	}, headroom.CompressOptions{Model: "gpt-4o"})

	out, err := c.GetMetrics(context.Background(), &headroom.MetricsQuery{Limit: 5})
	if err != nil {
		t.Fatalf("get metrics: %v", err)
	}
	// Metrics may be empty if proxy doesn't track recent_requests for /v1/compress.
	t.Logf("got %d metrics rows", len(out))
}

func TestParity_GetSummary(t *testing.T) {
	c := newTestClient(t)
	out, err := c.GetSummary(context.Background())
	if err != nil {
		t.Fatalf("get summary: %v", err)
	}
	if out == nil {
		t.Fatal("nil summary")
	}
}

func TestParity_GetStats(t *testing.T) {
	c := newTestClient(t)
	out, err := c.GetStats(context.Background())
	if err != nil {
		t.Fatalf("get stats: %v", err)
	}
	if out == nil {
		t.Fatal("nil stats")
	}
}

func TestParity_CCRStats(t *testing.T) {
	c := newTestClient(t)
	out, err := c.GetCCRStats(context.Background())
	if err != nil {
		t.Fatalf("ccr stats: %v", err)
	}
	if out == nil {
		t.Fatal("nil ccr stats")
	}
	if out.Store.MaxEntries == 0 {
		t.Errorf("expected store.max_entries > 0, got %+v", out.Store)
	}
}

func TestParity_RetrieveUnknownHash(t *testing.T) {
	c := newTestClient(t)
	// Unknown hash — should either return a structured error or an empty result.
	_, err := c.Retrieve(context.Background(), "nonexistent_hash_xyz")
	if err != nil {
		// Typed error is acceptable.
		if _, ok := headroom.AsCompressError(err); !ok {
			var herr *headroom.Error
			if !errorsAs(err, &herr) {
				t.Errorf("expected typed Headroom error, got %v", err)
			}
		}
	}
}

func TestParity_HandleToolCall(t *testing.T) {
	c := newTestClient(t)
	// Send a fake tool call — proxy should respond with a structured error or empty result.
	_, err := c.HandleToolCall(context.Background(), map[string]any{
		"id":   "call_x",
		"name": "headroom_retrieve",
		"input": map[string]any{
			"hash": "nonexistent",
		},
	}, "openai")
	// Either way, the SDK call wiring works if no transport-level error.
	if err != nil {
		t.Logf("tool call returned error (often expected for fake hash): %v", err)
	}
}

func TestParity_TelemetryGetStats(t *testing.T) {
	c := newTestClient(t)
	out, err := c.Telemetry.GetStats(context.Background())
	if err != nil {
		t.Fatalf("telemetry stats: %v", err)
	}
	if out == nil {
		t.Fatal("nil telemetry stats")
	}
}

func TestParity_TelemetryExport(t *testing.T) {
	c := newTestClient(t)
	out, err := c.Telemetry.Export(context.Background())
	if err != nil {
		t.Fatalf("telemetry export: %v", err)
	}
	if out == nil {
		t.Fatal("nil telemetry export")
	}
}

func TestParity_TelemetryGetTools(t *testing.T) {
	c := newTestClient(t)
	out, err := c.Telemetry.GetTools(context.Background())
	if err != nil {
		t.Fatalf("telemetry tools: %v", err)
	}
	if out == nil {
		t.Fatal("nil tools")
	}
}

func TestParity_FeedbackGetStats(t *testing.T) {
	c := newTestClient(t)
	out, err := c.Feedback.GetStats(context.Background())
	if err != nil {
		t.Fatalf("feedback stats: %v", err)
	}
	if out == nil {
		t.Fatal("nil feedback stats")
	}
}

func TestParity_FeedbackGetHints(t *testing.T) {
	c := newTestClient(t)
	out, err := c.Feedback.GetHints(context.Background(), "search")
	if err != nil {
		t.Logf("feedback hints (expected for unseen tool): %v", err)
		return
	}
	if out == nil {
		t.Fatal("nil hints")
	}
}

func TestParity_TOINGetStats(t *testing.T) {
	c := newTestClient(t)
	out, err := c.TOIN.GetStats(context.Background())
	if err != nil {
		t.Fatalf("toin stats: %v", err)
	}
	if out == nil {
		t.Fatal("nil toin stats")
	}
}

func TestParity_TOINGetPatterns(t *testing.T) {
	c := newTestClient(t)
	out, err := c.TOIN.GetPatterns(context.Background(), 10)
	if err != nil {
		t.Fatalf("toin patterns: %v", err)
	}
	t.Logf("got %d toin patterns", len(out))
}

func TestParity_CompressRaw(t *testing.T) {
	c := newTestClient(t)
	out, err := c.CompressRaw(context.Background(), map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		"model":    "gpt-4o",
	})
	if err != nil {
		t.Fatalf("compress raw: %v", err)
	}
	if out["tokens_before"] == nil {
		t.Errorf("expected tokens_before in raw response, got %v", out)
	}
}

func TestParity_ChatCompletionsSimulate(t *testing.T) {
	c := newTestClient(t)
	sim, err := c.ChatCompletionsSimulate(context.Background(), "gpt-4o", []headroom.Message{
		{Role: "user", Content: headroom.TextContent("Hello, how are you today?")},
	})
	if err != nil {
		t.Fatalf("chat simulate: %v", err)
	}
	if sim.TokensBefore < 0 {
		t.Errorf("unexpected tokens_before: %d", sim.TokensBefore)
	}
}

func TestParity_MessagesSimulate(t *testing.T) {
	c := newTestClient(t)
	sim, err := c.MessagesSimulate(context.Background(), "claude-sonnet-4-5-20250929", []headroom.Message{
		{Role: "user", Content: headroom.TextContent("Hello")},
	})
	if err != nil {
		t.Fatalf("messages simulate: %v", err)
	}
	if sim == nil {
		t.Fatal("nil simulation result")
	}
}

func TestParity_ChatCompletionsCreate_NoUpstream(t *testing.T) {
	c := newTestClient(t)
	// Without OPENAI_API_KEY, this hits the proxy's passthrough and returns
	// a 4xx/5xx (no upstream key). The point is the SDK reaches the proxy
	// and gets a typed error rather than blowing up at marshaling.
	_, err := c.ChatCompletionsCreate(context.Background(), map[string]any{
		"model":    "gpt-4o",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}, &headroom.HeadroomParams{Mode: headroom.ModeOptimize})
	if err == nil {
		// If the user happens to have OPENAI_API_KEY set, success is fine too.
		return
	}
	// Must be a typed error if it failed.
	if _, ok := headroom.AsCompressError(err); ok {
		return
	}
	var herr *headroom.Error
	if !errorsAs(err, &herr) {
		t.Errorf("expected typed Headroom error, got %v", err)
	}
}

func TestParity_AnthropicAdapter(t *testing.T) {
	c := newTestClient(t)
	msgs := []any{
		map[string]any{"role": "user", "content": "what is 2+2?"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "4"},
			map[string]any{"type": "tool_use", "id": "tu_1", "name": "calc", "input": map[string]any{}},
		}},
	}
	res, err := anthropic.Compress(context.Background(), msgs, headroom.CompressOptions{Client: c, Model: "claude-sonnet-4-5-20250929"})
	if err != nil {
		t.Fatalf("anthropic adapter: %v", err)
	}
	if len(res.Messages) == 0 {
		t.Error("expected messages back")
	}
}

func TestParity_GeminiAdapter(t *testing.T) {
	c := newTestClient(t)
	msgs := []any{
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "hello"}}},
		map[string]any{"role": "model", "parts": []any{map[string]any{"text": "hi there"}}},
	}
	res, err := gemini.Compress(context.Background(), msgs, headroom.CompressOptions{Client: c, Model: "gemini-2.0-flash"})
	if err != nil {
		t.Fatalf("gemini adapter: %v", err)
	}
	if len(res.Messages) == 0 {
		t.Error("expected messages back")
	}
}

func TestParity_VercelFormatRoundTrip(t *testing.T) {
	c := newTestClient(t)
	// Vercel-shape input — ensure detection routes through and proxy accepts.
	msgs := []any{
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "hello"},
		}},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "hi"},
			map[string]any{"type": "tool-call", "toolCallId": "tc_1", "toolName": "search", "input": map[string]any{"q": "go"}},
		}},
	}
	if format.Detect(msgs) != format.Vercel {
		t.Fatalf("expected vercel detection; got %s", format.Detect(msgs))
	}
	res, err := headroom.Compress(context.Background(), msgs, headroom.CompressOptions{Client: c, Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("vercel compress: %v", err)
	}
	if len(res.Messages) == 0 {
		t.Error("expected messages back")
	}
}

// TestParity_HooksFire verifies CompressionHooks methods are called.
func TestParity_HooksFire(t *testing.T) {
	c := newTestClient(t)
	h := &recordingHooks{}
	msgs := []any{map[string]any{"role": "user", "content": "Hello"}}
	_, err := headroom.Compress(context.Background(), msgs, headroom.CompressOptions{
		Client: c,
		Model:  "gpt-4o",
		Hooks:  h,
	})
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if !h.preCalled {
		t.Error("PreCompress not called")
	}
	if !h.biasesCalled {
		t.Error("ComputeBiases not called")
	}
	if !h.postCalled {
		t.Error("PostCompress not called")
	}
	if h.lastEvent.Model != "gpt-4o" {
		t.Errorf("event.Model: want gpt-4o, got %q", h.lastEvent.Model)
	}
}

func TestParity_HealthShape(t *testing.T) {
	c := newTestClient(t)
	h, err := c.Health(context.Background())
	if err != nil {
		t.Fatalf("health: %v", err)
	}
	if h.Version == "" {
		t.Error("version empty")
	}
	if !h.Config.Optimize {
		t.Error("expected optimize=true on default proxy")
	}
}

func TestParity_PrometheusMetricsContent(t *testing.T) {
	c := newTestClient(t)
	out, err := c.PrometheusMetrics(context.Background())
	if err != nil {
		t.Fatalf("prometheus: %v", err)
	}
	if !strings.Contains(out, "headroom") && !strings.Contains(out, "# HELP") {
		t.Errorf("prometheus output doesn't look like prom format; first 200 chars: %.200s", out)
	}
}

func TestParity_ProxyStatsShape(t *testing.T) {
	c := newTestClient(t)
	s, err := c.ProxyStats(context.Background())
	if err != nil {
		t.Fatalf("proxy stats: %v", err)
	}
	if s == nil {
		t.Fatal("nil stats")
	}
	// The shape is sufficiently rich that we just verify decoding didn't crash.
}

// TestParity_ConfigSentToProxy verifies that WithConfig is serialized into
// the request body.
func TestParity_ConfigSentToProxy(t *testing.T) {
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"messages":[{"role":"user","content":"hello world"}],
			"tokens_before":2,
			"tokens_after":2,
			"tokens_saved":0,
			"compression_ratio":1,
			"transforms_applied":[],
			"ccr_hashes":[]
		}`))
	}))
	defer server.Close()

	c := headroom.NewClient(
		headroom.WithBaseURL(server.URL),
		headroom.WithTimeout(15*time.Second),
		headroom.WithConfig(&headroom.Config{
			SmartCrusher: &headroom.SmartCrusherConfig{
				Enabled:          headroom.Bool(true),
				MinTokensToCrush: headroom.Int(50),
			},
		}),
	)
	res, err := c.Compress(context.Background(), []headroom.Message{
		{Role: "user", Content: headroom.TextContent("hello world")},
	}, headroom.CompressOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("compress with config: %v", err)
	}
	if res.TokensBefore == 0 {
		t.Error("expected non-zero tokens_before")
	}
	config, ok := requestBody["config"].(map[string]any)
	if !ok {
		t.Fatalf("expected config object in request, got %#v", requestBody["config"])
	}
	smartCrusher, ok := config["smart_crusher"].(map[string]any)
	if !ok || smartCrusher["enabled"] != true || smartCrusher["min_tokens_to_crush"] != float64(50) {
		t.Fatalf("unexpected smart_crusher config: %#v", smartCrusher)
	}
}

func TestParity_ChatCompletionsStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		if body["stream"] != true {
			t.Errorf("expected stream=true, got %#v", body["stream"])
		}
		if got := r.Header.Get("x-headroom-mode"); got != string(headroom.ModeOptimize) {
			t.Errorf("unexpected mode header: %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-1\"}\n\ndata: [DONE]\n\n"))
	}))
	defer server.Close()

	client := headroom.NewClient(headroom.WithBaseURL(server.URL))
	resp, err := client.ChatCompletionsStream(
		context.Background(),
		map[string]any{"model": "gpt-4o", "messages": []any{}},
		&headroom.HeadroomParams{Mode: headroom.ModeOptimize},
	)
	if err != nil {
		t.Fatalf("stream request: %v", err)
	}
	events, err := sdkstream.Collect(sdkstream.ParseSSE(resp))
	if err != nil {
		t.Fatalf("parse stream: %v", err)
	}
	if len(events) != 1 || events[0]["id"] != "chunk-1" {
		t.Fatalf("unexpected events: %#v", events)
	}
}

// TestParity_FormatRoundTripIdempotent verifies the format conversions reverse cleanly.
func TestParity_FormatRoundTripIdempotent(t *testing.T) {
	original := []any{
		map[string]any{"role": "user", "content": "hi"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "hello"},
			map[string]any{"type": "tool_use", "id": "tu_1", "name": "search", "input": map[string]any{"q": "go"}},
		}},
	}

	openai := format.ToOpenAI(original)
	back := format.FromOpenAI(openai, format.Anthropic)

	// We don't require exact equality (anthropic→openai→anthropic loses formatting),
	// just that the round-trip yields valid Anthropic-shape messages.
	if len(back) < 2 {
		t.Fatalf("round-trip lost messages: %d", len(back))
	}
	asst, _ := back[1].(map[string]any)
	if asst == nil || asst["role"] != "assistant" {
		t.Errorf("round-tripped assistant role missing: %+v", back[1])
	}
}

// recordingHooks captures hook calls.
type recordingHooks struct {
	preCalled    bool
	biasesCalled bool
	postCalled   bool
	lastEvent    headroom.CompressEvent
}

func (h *recordingHooks) PreCompress(_ context.Context, m []any, _ headroom.CompressContext) ([]any, error) {
	h.preCalled = true
	return m, nil
}
func (h *recordingHooks) ComputeBiases(_ context.Context, _ []headroom.Message, _ headroom.CompressContext) (map[int]float64, error) {
	h.biasesCalled = true
	return nil, nil
}
func (h *recordingHooks) PostCompress(_ context.Context, e headroom.CompressEvent) error {
	h.postCalled = true
	h.lastEvent = e
	return nil
}

// errorsAs is a local errors.As helper so we don't import "errors" in every test.
func errorsAs(err error, target any) bool {
	type asUnwrap interface{ As(any) bool }
	if v, ok := err.(asUnwrap); ok {
		return v.As(target)
	}
	// Fallback to type assertion via JSON of err — sufficient for typed Errors.
	if herr, ok := err.(*headroom.Error); ok {
		if t, ok := target.(**headroom.Error); ok {
			*t = herr
			return true
		}
	}
	_ = json.Unmarshal // keep import
	return false
}
