package headroom

import "testing"

func TestExtractUserQuery(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "system", "content": "sys"},
		map[string]any{"role": "user", "content": "first"},
		map[string]any{"role": "assistant", "content": "ok"},
		map[string]any{"role": "user", "content": "second"},
	}
	if got := ExtractUserQuery(msgs); got != "second" {
		t.Errorf("want 'second', got %q", got)
	}
}

func TestCountTurns(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "user", "content": "a"},
		map[string]any{"role": "assistant", "content": "b"},
		map[string]any{"role": "user", "content": "c"},
	}
	if n := CountTurns(msgs); n != 2 {
		t.Errorf("want 2, got %d", n)
	}
}

func TestExtractToolCalls(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "assistant", "tool_calls": []any{
			map[string]any{"id": "x", "type": "function", "function": map[string]any{"name": "search", "arguments": "{}"}},
		}},
	}
	got := ExtractToolCalls(msgs)
	if len(got) != 1 || got[0] != "search" {
		t.Errorf("want [search], got %v", got)
	}
}
