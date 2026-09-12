// Package format provides message-format detection and conversion.
//
// Supports four formats, all reduced to OpenAI for the proxy:
//   - OpenAI:    {role, content, tool_calls?, tool_call_id?}
//   - Anthropic: {role, content (string | block[])} with tool_use/tool_result
//   - Vercel:    {role, content (string | part[])} with tool-call/tool-result
//   - Gemini:    {role: model|user, parts[]} with functionCall/functionResponse
package format

import (
	"encoding/json"
	"fmt"
)

type MessageFormat string

const (
	OpenAI    MessageFormat = "openai"
	Anthropic MessageFormat = "anthropic"
	Vercel    MessageFormat = "vercel"
	Gemini    MessageFormat = "gemini"
)

// Detect inspects the messages slice and returns the detected format.
// Falls back to OpenAI for plain {role, content: string} messages.
func Detect(messages []any) MessageFormat {
	for _, raw := range messages {
		m := asMap(raw)
		if m == nil {
			continue
		}
		_, hasParts := m["parts"]
		_, hasContent := m["content"]
		if hasParts && !hasContent {
			return Gemini
		}
		if m["role"] == "model" {
			return Gemini
		}
		if _, ok := m["tool_calls"]; ok && m["role"] == "assistant" {
			return OpenAI
		}
		if m["role"] == "tool" {
			if _, ok := m["tool_call_id"]; ok {
				if _, isString := m["content"].(string); isString {
					return OpenAI
				}
			}
		}
		if parts, ok := m["content"].([]any); ok {
			for _, p := range parts {
				pm := asMap(p)
				if pm == nil {
					continue
				}
				t, _ := pm["type"].(string)
				switch t {
				case "tool-call", "tool-result":
					return Vercel
				case "tool_use", "tool_result":
					return Anthropic
				case "image":
					if src, ok := pm["source"].(map[string]any); ok {
						if _, ok := src["type"]; ok {
							return Anthropic
						}
					}
				}
			}
		}
	}
	return OpenAI
}

// ToOpenAI converts any-format messages to OpenAI shape.
func ToOpenAI(messages []any) []any {
	switch Detect(messages) {
	case Anthropic:
		return anthropicToOpenAI(messages)
	case Vercel:
		return vercelToOpenAI(messages)
	case Gemini:
		return geminiToOpenAI(messages)
	default:
		return messages
	}
}

// FromOpenAI converts OpenAI-shape messages back to the target format.
func FromOpenAI(messages []any, target MessageFormat) []any {
	switch target {
	case Anthropic:
		return openAIToAnthropic(messages)
	case Vercel:
		return openAIToVercel(messages)
	case Gemini:
		return openAIToGemini(messages)
	default:
		return messages
	}
}

// ----- Anthropic <-> OpenAI -----

func anthropicToOpenAI(messages []any) []any {
	out := make([]any, 0, len(messages))
	for _, raw := range messages {
		m := asMap(raw)
		if m == nil {
			continue
		}
		role, _ := m["role"].(string)
		switch role {
		case "user":
			if s, ok := m["content"].(string); ok {
				out = append(out, map[string]any{"role": "user", "content": s})
				continue
			}
			if blocks, ok := m["content"].([]any); ok {
				var texts []string
				var toolResults []map[string]any
				for _, b := range blocks {
					bm := asMap(b)
					if bm == nil {
						continue
					}
					switch bm["type"] {
					case "text":
						if t, ok := bm["text"].(string); ok {
							texts = append(texts, t)
						}
					case "tool_result":
						toolResults = append(toolResults, bm)
					}
				}
				if len(texts) > 0 {
					out = append(out, map[string]any{"role": "user", "content": joinStrings(texts, "\n")})
				}
				for _, tr := range toolResults {
					var contentStr string
					switch c := tr["content"].(type) {
					case string:
						contentStr = c
					case []any:
						var parts []string
						for _, b := range c {
							bm := asMap(b)
							if bm == nil {
								continue
							}
							if t, ok := bm["text"].(string); ok {
								parts = append(parts, t)
							} else {
								j, _ := json.Marshal(bm)
								parts = append(parts, string(j))
							}
						}
						contentStr = joinStrings(parts, "\n")
					default:
						j, _ := json.Marshal(c)
						contentStr = string(j)
					}
					out = append(out, map[string]any{
						"role":         "tool",
						"content":      contentStr,
						"tool_call_id": tr["tool_use_id"],
					})
				}
			}
		case "assistant":
			if s, ok := m["content"].(string); ok {
				out = append(out, map[string]any{"role": "assistant", "content": s})
				continue
			}
			if blocks, ok := m["content"].([]any); ok {
				var texts []string
				var toolUses []map[string]any
				for _, b := range blocks {
					bm := asMap(b)
					if bm == nil {
						continue
					}
					switch bm["type"] {
					case "text":
						if t, ok := bm["text"].(string); ok {
							texts = append(texts, t)
						}
					case "tool_use":
						toolUses = append(toolUses, bm)
					}
				}
				msg := map[string]any{"role": "assistant"}
				if len(texts) > 0 {
					msg["content"] = joinStrings(texts, "\n")
				} else {
					msg["content"] = nil
				}
				if len(toolUses) > 0 {
					tcs := make([]map[string]any, 0, len(toolUses))
					for _, tu := range toolUses {
						args, _ := json.Marshal(tu["input"])
						tcs = append(tcs, map[string]any{
							"id":   tu["id"],
							"type": "function",
							"function": map[string]any{
								"name":      tu["name"],
								"arguments": string(args),
							},
						})
					}
					msg["tool_calls"] = tcs
				}
				out = append(out, msg)
			}
		}
	}
	return out
}

func openAIToAnthropic(messages []any) []any {
	out := make([]any, 0, len(messages))
	for _, raw := range messages {
		m := asMap(raw)
		if m == nil {
			continue
		}
		role, _ := m["role"].(string)
		switch role {
		case "system":
			out = append(out, map[string]any{"role": "user", "content": m["content"]})
		case "user":
			out = append(out, map[string]any{"role": "user", "content": m["content"]})
		case "assistant":
			blocks := []map[string]any{}
			if s, ok := m["content"].(string); ok && s != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": s})
			}
			if tcs, ok := m["tool_calls"].([]any); ok {
				for _, tc := range tcs {
					tcm := asMap(tc)
					if tcm == nil {
						continue
					}
					fn := asMap(tcm["function"])
					var input any
					if fn != nil {
						_ = json.Unmarshal([]byte(fmt.Sprint(fn["arguments"])), &input)
					}
					blocks = append(blocks, map[string]any{
						"type":  "tool_use",
						"id":    tcm["id"],
						"name":  fn["name"],
						"input": input,
					})
				}
			}
			if len(blocks) == 1 && blocks[0]["type"] == "text" {
				out = append(out, map[string]any{"role": "assistant", "content": blocks[0]["text"]})
			} else {
				bs := make([]any, len(blocks))
				for i, b := range blocks {
					bs[i] = b
				}
				out = append(out, map[string]any{"role": "assistant", "content": bs})
			}
		case "tool":
			out = append(out, map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]},
				},
			})
		}
	}
	return out
}

// ----- Vercel <-> OpenAI -----

func vercelToOpenAI(messages []any) []any {
	out := make([]any, 0, len(messages))
	for _, raw := range messages {
		m := asMap(raw)
		if m == nil {
			continue
		}
		role, _ := m["role"].(string)
		switch role {
		case "system":
			out = append(out, map[string]any{"role": "system", "content": stringify(m["content"])})
		case "user":
			parts, _ := m["content"].([]any)
			if parts == nil {
				if s, ok := m["content"].(string); ok {
					out = append(out, map[string]any{"role": "user", "content": s})
					continue
				}
			}
			texts := []string{}
			hasImage := false
			for _, p := range parts {
				pm := asMap(p)
				if pm == nil {
					continue
				}
				switch pm["type"] {
				case "text":
					if s, ok := pm["text"].(string); ok {
						texts = append(texts, s)
					}
				case "image":
					hasImage = true
				}
			}
			if !hasImage {
				out = append(out, map[string]any{"role": "user", "content": joinStrings(texts, "")})
			} else {
				openaiParts := []any{}
				for _, p := range parts {
					pm := asMap(p)
					if pm == nil {
						continue
					}
					switch pm["type"] {
					case "text":
						openaiParts = append(openaiParts, map[string]any{"type": "text", "text": pm["text"]})
					case "image":
						url := stringify(pm["image"])
						openaiParts = append(openaiParts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
					}
				}
				out = append(out, map[string]any{"role": "user", "content": openaiParts})
			}
		case "assistant":
			if s, ok := m["content"].(string); ok {
				out = append(out, map[string]any{"role": "assistant", "content": s})
				continue
			}
			parts, _ := m["content"].([]any)
			texts := []string{}
			toolCalls := []map[string]any{}
			for _, p := range parts {
				pm := asMap(p)
				if pm == nil {
					continue
				}
				switch pm["type"] {
				case "text":
					if s, ok := pm["text"].(string); ok {
						texts = append(texts, s)
					}
				case "tool-call":
					input := pm["input"]
					if input == nil {
						input = pm["args"]
					}
					argsBytes, _ := json.Marshal(input)
					toolCalls = append(toolCalls, map[string]any{
						"id":   pm["toolCallId"],
						"type": "function",
						"function": map[string]any{
							"name":      pm["toolName"],
							"arguments": string(argsBytes),
						},
					})
				}
			}
			msg := map[string]any{"role": "assistant"}
			if len(texts) > 0 {
				msg["content"] = joinStrings(texts, "")
			} else {
				msg["content"] = nil
			}
			if len(toolCalls) > 0 {
				ts := make([]any, len(toolCalls))
				for i, t := range toolCalls {
					ts[i] = t
				}
				msg["tool_calls"] = ts
			}
			out = append(out, msg)
		case "tool":
			parts, _ := m["content"].([]any)
			for _, p := range parts {
				pm := asMap(p)
				if pm == nil || pm["type"] != "tool-result" {
					continue
				}
				var contentStr string
				if output, ok := pm["output"].(map[string]any); ok {
					val := output["value"]
					if val == nil {
						val = output
					}
					if s, ok := val.(string); ok {
						contentStr = s
					} else {
						b, _ := json.Marshal(val)
						contentStr = string(b)
					}
				} else if r := pm["result"]; r != nil {
					if s, ok := r.(string); ok {
						contentStr = s
					} else {
						b, _ := json.Marshal(r)
						contentStr = string(b)
					}
				}
				out = append(out, map[string]any{
					"role":         "tool",
					"content":      contentStr,
					"tool_call_id": pm["toolCallId"],
				})
			}
		}
	}
	return out
}

func openAIToVercel(messages []any) []any {
	out := make([]any, 0, len(messages))
	for _, raw := range messages {
		m := asMap(raw)
		if m == nil {
			continue
		}
		role, _ := m["role"].(string)
		switch role {
		case "system":
			out = append(out, map[string]any{"role": "system", "content": m["content"]})
		case "user":
			if s, ok := m["content"].(string); ok {
				out = append(out, map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": s}}})
			} else if parts, ok := m["content"].([]any); ok {
				ps := []any{}
				for _, p := range parts {
					pm := asMap(p)
					if pm == nil {
						continue
					}
					switch pm["type"] {
					case "text":
						ps = append(ps, map[string]any{"type": "text", "text": pm["text"]})
					case "image_url":
						img, _ := pm["image_url"].(map[string]any)
						if img != nil {
							ps = append(ps, map[string]any{"type": "image", "image": img["url"]})
						}
					}
				}
				out = append(out, map[string]any{"role": "user", "content": ps})
			}
		case "assistant":
			ps := []any{}
			if s, ok := m["content"].(string); ok && s != "" {
				ps = append(ps, map[string]any{"type": "text", "text": s})
			}
			if tcs, ok := m["tool_calls"].([]any); ok {
				for _, tc := range tcs {
					tcm := asMap(tc)
					if tcm == nil {
						continue
					}
					fn := asMap(tcm["function"])
					var input any
					if fn != nil {
						if err := json.Unmarshal([]byte(fmt.Sprint(fn["arguments"])), &input); err != nil {
							input = map[string]any{}
						}
					}
					ps = append(ps, map[string]any{
						"type":       "tool-call",
						"toolCallId": tcm["id"],
						"toolName":   fn["name"],
						"input":      input,
					})
				}
			}
			out = append(out, map[string]any{"role": "assistant", "content": ps})
		case "tool":
			var parsed any
			if s, ok := m["content"].(string); ok {
				if err := json.Unmarshal([]byte(s), &parsed); err != nil {
					parsed = s
				}
			}
			var output map[string]any
			if s, ok := parsed.(string); ok {
				output = map[string]any{"type": "text", "value": s}
			} else {
				output = map[string]any{"type": "json", "value": parsed}
			}
			out = append(out, map[string]any{
				"role": "tool",
				"content": []any{
					map[string]any{
						"type":       "tool-result",
						"toolCallId": m["tool_call_id"],
						"toolName":   "unknown",
						"output":     output,
					},
				},
			})
		}
	}
	return out
}

// ----- Gemini <-> OpenAI -----

func geminiToOpenAI(messages []any) []any {
	out := make([]any, 0, len(messages))
	for _, raw := range messages {
		m := asMap(raw)
		if m == nil {
			continue
		}
		role := "user"
		if r, _ := m["role"].(string); r == "model" {
			role = "assistant"
		}
		parts, _ := m["parts"].([]any)
		switch role {
		case "user":
			texts := []string{}
			funcResponses := []map[string]any{}
			for _, p := range parts {
				pm := asMap(p)
				if pm == nil {
					continue
				}
				if t, ok := pm["text"]; ok && t != nil {
					if s, ok := t.(string); ok {
						texts = append(texts, s)
					}
				}
				if fr, ok := pm["functionResponse"].(map[string]any); ok {
					funcResponses = append(funcResponses, fr)
				}
			}
			if len(texts) > 0 {
				out = append(out, map[string]any{"role": "user", "content": joinStrings(texts, "\n")})
			}
			for _, fr := range funcResponses {
				resp, _ := json.Marshal(fr["response"])
				name, _ := fr["name"].(string)
				out = append(out, map[string]any{
					"role":         "tool",
					"content":      string(resp),
					"tool_call_id": "gemini_" + name,
				})
			}
		case "assistant":
			texts := []string{}
			funcCalls := []map[string]any{}
			for _, p := range parts {
				pm := asMap(p)
				if pm == nil {
					continue
				}
				if t, ok := pm["text"]; ok && t != nil {
					if s, ok := t.(string); ok {
						texts = append(texts, s)
					}
				}
				if fc, ok := pm["functionCall"].(map[string]any); ok {
					funcCalls = append(funcCalls, fc)
				}
			}
			msg := map[string]any{"role": "assistant"}
			if len(texts) > 0 {
				msg["content"] = joinStrings(texts, "\n")
			} else {
				msg["content"] = nil
			}
			if len(funcCalls) > 0 {
				tcs := make([]any, 0, len(funcCalls))
				for _, fc := range funcCalls {
					name, _ := fc["name"].(string)
					args, _ := json.Marshal(fc["args"])
					tcs = append(tcs, map[string]any{
						"id":   "gemini_" + name,
						"type": "function",
						"function": map[string]any{
							"name":      name,
							"arguments": string(args),
						},
					})
				}
				msg["tool_calls"] = tcs
			}
			out = append(out, msg)
		}
	}
	return out
}

func openAIToGemini(messages []any) []any {
	out := make([]any, 0, len(messages))
	for _, raw := range messages {
		m := asMap(raw)
		if m == nil {
			continue
		}
		role, _ := m["role"].(string)
		switch role {
		case "system", "user":
			text := ""
			if s, ok := m["content"].(string); ok {
				text = s
			} else if parts, ok := m["content"].([]any); ok {
				ts := []string{}
				for _, p := range parts {
					pm := asMap(p)
					if pm == nil {
						continue
					}
					if pm["type"] == "text" {
						if s, ok := pm["text"].(string); ok {
							ts = append(ts, s)
						}
					}
				}
				text = joinStrings(ts, "\n")
			}
			out = append(out, map[string]any{"role": "user", "parts": []any{map[string]any{"text": text}}})
		case "assistant":
			parts := []any{}
			if s, ok := m["content"].(string); ok && s != "" {
				parts = append(parts, map[string]any{"text": s})
			}
			if tcs, ok := m["tool_calls"].([]any); ok {
				for _, tc := range tcs {
					tcm := asMap(tc)
					if tcm == nil {
						continue
					}
					fn := asMap(tcm["function"])
					var args any
					if fn != nil {
						_ = json.Unmarshal([]byte(fmt.Sprint(fn["arguments"])), &args)
					}
					parts = append(parts, map[string]any{
						"functionCall": map[string]any{
							"name": fn["name"],
							"args": args,
						},
					})
				}
			}
			out = append(out, map[string]any{"role": "model", "parts": parts})
		case "tool":
			var resp any
			if s, ok := m["content"].(string); ok {
				if err := json.Unmarshal([]byte(s), &resp); err != nil {
					resp = map[string]any{"result": s}
				}
			}
			name := ""
			if s, ok := m["tool_call_id"].(string); ok {
				name = stripPrefix(s, "gemini_")
			}
			out = append(out, map[string]any{
				"role": "user",
				"parts": []any{
					map[string]any{"functionResponse": map[string]any{"name": name, "response": resp}},
				},
			})
		}
	}
	return out
}

// helpers

func asMap(v any) map[string]any {
	if v == nil {
		return nil
	}
	if m, ok := v.(map[string]any); ok {
		return m
	}
	// Round-trip via JSON to handle struct values.
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil
	}
	return m
}

func joinStrings(in []string, sep string) string {
	switch len(in) {
	case 0:
		return ""
	case 1:
		return in[0]
	}
	n := len(sep) * (len(in) - 1)
	for _, s := range in {
		n += len(s)
	}
	b := make([]byte, 0, n)
	for i, s := range in {
		if i > 0 {
			b = append(b, sep...)
		}
		b = append(b, s...)
	}
	return string(b)
}

func stringify(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	return string(b)
}

func stripPrefix(s, prefix string) string {
	if len(s) >= len(prefix) && s[:len(prefix)] == prefix {
		return s[len(prefix):]
	}
	return s
}
