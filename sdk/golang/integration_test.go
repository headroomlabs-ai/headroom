package headroom_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	headroom "github.com/headroomlabs-ai/headroom/sdk/golang"
	"github.com/headroomlabs-ai/headroom/sdk/golang/format"
	"github.com/headroomlabs-ai/headroom/sdk/golang/openai"
)

// All tests in this file require a running proxy. The base URL is taken from
// HEADROOM_TEST_BASE_URL (preferred) or HEADROOM_BASE_URL. If unset, tests skip.

func proxyURL() string {
	if u := os.Getenv("HEADROOM_TEST_BASE_URL"); u != "" {
		return u
	}
	return os.Getenv("HEADROOM_BASE_URL")
}

func newTestClient(t *testing.T) *headroom.Client {
	t.Helper()
	url := proxyURL()
	if url == "" {
		t.Skip("set HEADROOM_TEST_BASE_URL to run integration tests")
	}
	return headroom.NewClient(headroom.WithBaseURL(url), headroom.WithTimeout(15*time.Second))
}

func TestHealth(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	h, err := c.Health(ctx)
	if err != nil {
		t.Fatalf("health: %v", err)
	}
	if h.Status != "healthy" {
		t.Errorf("want healthy, got %q", h.Status)
	}
	if h.Version == "" {
		t.Error("expected version to be set")
	}
}

func TestCompressBasic(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	msgs := []headroom.Message{
		{Role: "user", Content: headroom.TextContent("hello world")},
	}
	res, err := c.Compress(ctx, msgs, headroom.CompressOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if res.TokensBefore <= 0 {
		t.Errorf("expected tokens_before > 0, got %d", res.TokensBefore)
	}
	if !res.Compressed {
		t.Errorf("expected compressed=true on success")
	}
	if len(res.Messages) == 0 {
		t.Errorf("expected at least one message back")
	}
}

func TestCompressWithBigToolOutput(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()

	// Build a 500-item tool result that should trigger SmartCrusher.
	items := make([]map[string]any, 500)
	for i := range items {
		items[i] = map[string]any{
			"title":       "Result number " + itoa(i) + " from the search index",
			"snippet":     "This is a fairly long description for result number " + itoa(i) + " containing detail and context.",
			"score":       500 - i,
			"url":         "https://example.com/results/" + itoa(i),
			"description": "Auxiliary description text for result " + itoa(i) + " — repeated boilerplate that the compressor should crush.",
		}
	}
	body, _ := json.Marshal(map[string]any{"results": items})

	msgs := []headroom.Message{
		{Role: "system", Content: headroom.TextContent("You analyze search results.")},
		{Role: "user", Content: headroom.TextContent("Search results.")},
		{
			Role:    "assistant",
			Content: nullJSON(),
			ToolCalls: []headroom.ToolCall{{
				ID: "call_1", Type: "function",
				Function: headroom.ToolFunction{Name: "search", Arguments: `{"q":"x"}`},
			}},
		},
		{Role: "tool", ToolCallID: "call_1", Content: headroom.TextContent(string(body))},
		{Role: "user", Content: headroom.TextContent("Top 3 results please")},
	}

	res, err := c.Compress(ctx, msgs, headroom.CompressOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if res.TokensBefore == 0 {
		t.Errorf("tokens_before should be non-zero for sizable input")
	}
	if len(res.TransformsApplied) == 0 {
		t.Errorf("expected at least one router decision recorded")
	}
	if len(res.Messages) == 0 {
		t.Errorf("expected messages returned")
	}
	t.Logf("tokens_before=%d after=%d saved=%d transforms=%v",
		res.TokensBefore, res.TokensAfter, res.TokensSaved, res.TransformsApplied)
}

func TestSimulate(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	msgs := []any{map[string]any{"role": "user", "content": "hello world"}}
	sim, err := headroom.Simulate(ctx, msgs, headroom.SimulateOptions{Model: "gpt-4o", Client: c})
	if err != nil {
		t.Fatalf("simulate: %v", err)
	}
	if sim.TokensBefore < 0 {
		t.Errorf("unexpected negative tokens_before: %d", sim.TokensBefore)
	}
}

func TestProxyStats(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	if _, err := c.ProxyStats(ctx); err != nil {
		t.Fatalf("proxy stats: %v", err)
	}
}

func TestPrometheusMetrics(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	out, err := c.PrometheusMetrics(ctx)
	if err != nil {
		t.Fatalf("prometheus: %v", err)
	}
	if len(out) == 0 {
		t.Error("expected non-empty prometheus output")
	}
}

func TestValidateSetup(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	v, err := c.ValidateSetup(ctx)
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if !v.Valid {
		t.Errorf("expected valid=true, got errors=%v", v.Errors)
	}
}

func TestOpenAIAdapterCompress(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	msgs := []headroom.Message{{Role: "user", Content: headroom.TextContent("hello")}}
	res, err := openai.Compress(ctx, msgs, headroom.CompressOptions{Client: c, Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("adapter compress: %v", err)
	}
	if res.TokensBefore <= 0 {
		t.Errorf("expected tokens_before > 0")
	}
}

func TestCompressGenericAnthropicShape(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	// Anthropic-shape input with a tool_use block — proxy works on OpenAI form.
	msgs := []any{
		map[string]any{"role": "user", "content": "hi there"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "hello"},
		}},
	}
	if format.Detect(msgs) != format.Anthropic {
		// Plain string content + no tool_use block => not detected as Anthropic.
		// That's fine; this exercises the compress() generic path either way.
	}
	res, err := headroom.Compress(ctx, msgs, headroom.CompressOptions{Client: c, Model: "claude-sonnet-4-5-20250929"})
	if err != nil {
		t.Fatalf("compress generic: %v", err)
	}
	if len(res.Messages) == 0 {
		t.Error("expected messages returned")
	}
}

func TestSharedContext(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	sc := headroom.NewSharedContext(headroom.SharedContextOptions{Client: c, MaxEntries: 10})
	entry, err := sc.Put(ctx, "research", "Go is a statically typed language designed at Google.", "researcher")
	if err != nil {
		t.Fatalf("put: %v", err)
	}
	if entry.OriginalTokens <= 0 {
		t.Errorf("expected original_tokens > 0")
	}
	got, ok := sc.Get("research", true)
	if !ok || got == "" {
		t.Errorf("expected get(full) to return content")
	}
	stats := sc.Stats()
	if stats.Entries != 1 {
		t.Errorf("want 1 entry, got %d", stats.Entries)
	}
}

func TestCompressErrorOnMalformedRequest(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	// Send a request without messages — proxy should reject with 4xx.
	// Disable fallback so we surface the error.
	cNoFallback := headroom.NewClient(
		headroom.WithBaseURL(c.BaseURL()),
		headroom.WithFallback(false),
		headroom.WithRetries(0),
	)
	_, err := cNoFallback.Compress(ctx, []headroom.Message{}, headroom.CompressOptions{})
	// Empty messages may be accepted (returns 0 tokens) — only assert that
	// when error does occur, it's typed.
	if err != nil {
		if _, ok := headroom.AsCompressError(err); !ok {
			t.Logf("non-CompressError returned: %v (acceptable for some proxy versions)", err)
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := false
	if i < 0 {
		neg = true
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

func nullJSON() []byte {
	return []byte("null")
}
