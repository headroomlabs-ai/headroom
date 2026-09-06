package headroom

import (
	"context"
	"encoding/json"
)

// CompressContext is passed to hooks.
type CompressContext struct {
	Model      string
	UserQuery  string
	TurnNumber int
	ToolCalls  []string
	Provider   string
}

// CompressEvent is observable post-compression result handed to PostCompress.
type CompressEvent struct {
	TokensBefore      int
	TokensAfter       int
	TokensSaved       int
	CompressionRatio  float64
	TransformsApplied []string
	CCRHashes         []string
	Model             string
	UserQuery         string
	Provider          string
}

// Hooks customizes compression. Implement any subset; defaults are no-ops.
// All methods receive context.Context for cancellation.
type Hooks interface {
	PreCompress(ctx context.Context, messages []any, c CompressContext) ([]any, error)
	ComputeBiases(ctx context.Context, messages []Message, c CompressContext) (map[int]float64, error)
	PostCompress(ctx context.Context, event CompressEvent) error
}

// NoopHooks is a Hooks implementation that does nothing — embed it to override
// only the methods you care about.
type NoopHooks struct{}

func (NoopHooks) PreCompress(_ context.Context, m []any, _ CompressContext) ([]any, error) {
	return m, nil
}
func (NoopHooks) ComputeBiases(_ context.Context, _ []Message, _ CompressContext) (map[int]float64, error) {
	return nil, nil
}
func (NoopHooks) PostCompress(_ context.Context, _ CompressEvent) error { return nil }

// Hook helpers — mirror TS extractUserQuery / countTurns / extractToolCalls.

// ExtractUserQuery returns the last user message text from a messages slice
// (any format). Returns "" if none found.
func ExtractUserQuery(messages []any) string {
	for i := len(messages) - 1; i >= 0; i-- {
		m, ok := messages[i].(map[string]any)
		if !ok {
			continue
		}
		if m["role"] != "user" {
			continue
		}
		switch c := m["content"].(type) {
		case string:
			return c
		case []any:
			for _, p := range c {
				pm, ok := p.(map[string]any)
				if !ok {
					continue
				}
				if t, _ := pm["type"].(string); t == "text" {
					if s, _ := pm["text"].(string); s != "" {
						return s
					}
				}
				if s, _ := pm["text"].(string); s != "" {
					return s
				}
			}
		}
	}
	return ""
}

// CountTurns returns the number of user messages.
func CountTurns(messages []any) int {
	n := 0
	for _, m := range messages {
		if mm, ok := m.(map[string]any); ok && mm["role"] == "user" {
			n++
		}
	}
	return n
}

// ExtractToolCalls returns the list of tool-call names referenced in messages.
func ExtractToolCalls(messages []any) []string {
	var out []string
	for _, m := range messages {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if tcs, ok := mm["tool_calls"].([]any); ok {
			for _, tc := range tcs {
				if t, ok := tc.(map[string]any); ok {
					if fn, ok := t["function"].(map[string]any); ok {
						if name, _ := fn["name"].(string); name != "" {
							out = append(out, name)
							continue
						}
					}
					if name, _ := t["name"].(string); name != "" {
						out = append(out, name)
					}
				}
			}
		}
		if parts, ok := mm["content"].([]any); ok {
			for _, p := range parts {
				if pm, ok := p.(map[string]any); ok {
					if t, _ := pm["type"].(string); t == "tool_use" {
						if name, _ := pm["name"].(string); name != "" {
							out = append(out, name)
						}
					}
					if t, _ := pm["type"].(string); t == "tool-call" {
						if name, _ := pm["toolName"].(string); name != "" {
							out = append(out, name)
						}
					}
				}
			}
		}
	}
	return out
}

// messagesAsAny converts []Message to []any for hook consumption.
func messagesAsAny(in []Message) []any {
	out := make([]any, len(in))
	for i, m := range in {
		// Round-trip via JSON to give hooks a uniform map[string]any view.
		b, _ := json.Marshal(m)
		var v any
		_ = json.Unmarshal(b, &v)
		out[i] = v
	}
	return out
}

// anyToMessages converts []any back to []Message.
func anyToMessages(in []any) ([]Message, error) {
	out := make([]Message, len(in))
	for i, v := range in {
		b, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, &out[i]); err != nil {
			return nil, err
		}
	}
	return out, nil
}
