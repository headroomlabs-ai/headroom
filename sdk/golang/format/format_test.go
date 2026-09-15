package format

import (
	"reflect"
	"testing"
)

func TestDetectOpenAI(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "user", "content": "hello"},
		map[string]any{"role": "assistant", "content": "hi"},
	}
	if f := Detect(msgs); f != OpenAI {
		t.Errorf("want openai, got %s", f)
	}
}

func TestDetectAnthropic(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "tool_use", "id": "x", "name": "search", "input": map[string]any{}},
		}},
	}
	if f := Detect(msgs); f != Anthropic {
		t.Errorf("want anthropic, got %s", f)
	}
}

func TestDetectVercel(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "tool-call", "toolCallId": "x", "toolName": "y", "input": map[string]any{}},
		}},
	}
	if f := Detect(msgs); f != Vercel {
		t.Errorf("want vercel, got %s", f)
	}
}

func TestDetectGemini(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "hi"}}},
	}
	if f := Detect(msgs); f != Gemini {
		t.Errorf("want gemini, got %s", f)
	}
}

func TestAnthropicRoundTrip(t *testing.T) {
	in := []any{
		map[string]any{"role": "user", "content": "hello"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "hi there"},
			map[string]any{"type": "tool_use", "id": "tu_1", "name": "search", "input": map[string]any{"q": "go"}},
		}},
	}
	openai := ToOpenAI(in)
	if len(openai) != 2 {
		t.Fatalf("want 2 openai messages, got %d", len(openai))
	}
	asst := openai[1].(map[string]any)
	tcs, _ := asst["tool_calls"].([]map[string]any)
	if len(tcs) != 1 {
		t.Errorf("want 1 tool call, got %d", len(tcs))
	}
}

func TestVercelToolResultV6(t *testing.T) {
	in := []any{
		map[string]any{"role": "tool", "content": []any{
			map[string]any{
				"type":       "tool-result",
				"toolCallId": "tc_1",
				"toolName":   "search",
				"output":     map[string]any{"type": "json", "value": map[string]any{"results": []any{1, 2}}},
			},
		}},
	}
	openai := ToOpenAI(in)
	if len(openai) != 1 {
		t.Fatalf("want 1 message, got %d", len(openai))
	}
	tool := openai[0].(map[string]any)
	if tool["role"] != "tool" {
		t.Errorf("want tool role, got %v", tool["role"])
	}
	if tool["tool_call_id"] != "tc_1" {
		t.Errorf("want tool_call_id tc_1, got %v", tool["tool_call_id"])
	}
}

func TestGeminiRoundTrip(t *testing.T) {
	in := []any{
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "hi"}}},
		map[string]any{"role": "model", "parts": []any{
			map[string]any{"text": "hello"},
			map[string]any{"functionCall": map[string]any{"name": "search", "args": map[string]any{"q": "go"}}},
		}},
	}
	openai := ToOpenAI(in)
	if len(openai) != 2 {
		t.Fatalf("want 2 messages, got %d", len(openai))
	}
	asst := openai[1].(map[string]any)
	if asst["role"] != "assistant" {
		t.Errorf("want assistant, got %v", asst["role"])
	}
	tcs, _ := asst["tool_calls"].([]any)
	if len(tcs) != 1 {
		t.Errorf("want 1 tool call, got %d", len(tcs))
	}
}

func TestPassthroughOpenAI(t *testing.T) {
	in := []any{map[string]any{"role": "user", "content": "x"}}
	out := ToOpenAI(in)
	if !reflect.DeepEqual(in, out) {
		t.Errorf("openai passthrough should be identity")
	}
}
