package headroom

import "encoding/json"

// Message is the canonical OpenAI-style chat message used on the wire.
// Content is intentionally json.RawMessage so callers can pass strings,
// content-part arrays, or null (for assistant tool-call messages).
type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// TextContent builds a Message.Content holding a plain string.
func TextContent(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// JSONContent builds a Message.Content from any value (e.g. content parts).
func JSONContent(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

// CompressResult is what compress() returns.
type CompressResult struct {
	Messages          []Message `json:"messages"`
	TokensBefore      int       `json:"tokens_before"`
	TokensAfter       int       `json:"tokens_after"`
	TokensSaved       int       `json:"tokens_saved"`
	CompressionRatio  float64   `json:"compression_ratio"`
	TransformsApplied []string  `json:"transforms_applied"`
	CCRHashes         []string  `json:"ccr_hashes"`
	Compressed        bool      `json:"-"`
}

// CompressResultGeneric is like CompressResult but messages are kept in the
// caller's original (non-OpenAI) format.
type CompressResultGeneric struct {
	Messages          []any    `json:"messages"`
	TokensBefore      int      `json:"tokens_before"`
	TokensAfter       int      `json:"tokens_after"`
	TokensSaved       int      `json:"tokens_saved"`
	CompressionRatio  float64  `json:"compression_ratio"`
	TransformsApplied []string `json:"transforms_applied"`
	CCRHashes         []string `json:"ccr_hashes"`
	Compressed        bool     `json:"-"`
}

// proxyCompressResponse is the wire shape for /v1/compress.
type proxyCompressResponse struct {
	Messages          []Message `json:"messages"`
	TokensBefore      int       `json:"tokens_before"`
	TokensAfter       int       `json:"tokens_after"`
	TokensSaved       int       `json:"tokens_saved"`
	CompressionRatio  float64   `json:"compression_ratio"`
	TransformsApplied []string  `json:"transforms_applied"`
	CCRHashes         []string  `json:"ccr_hashes"`
}

type proxyErrorResponse struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}
