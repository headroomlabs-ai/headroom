package headroom

import (
	"context"
	"encoding/json"
	"net/http"
)

// ChatCompletionsCreate posts to /v1/chat/completions (OpenAI-style passthrough
// with automatic compression). Returns the raw decoded JSON; callers know
// what shape to expect from their model.
//
// For streaming, use ChatCompletionsStream.
func (c *Client) ChatCompletionsCreate(ctx context.Context, params map[string]any, hp *HeadroomParams) (map[string]any, error) {
	headers := map[string]string{}
	if hp != nil && hp.Mode != "" {
		headers["x-headroom-mode"] = string(hp.Mode)
	}
	if k := c.ProviderAPIKey("OPENAI_API_KEY"); k != "" {
		headers["Authorization"] = "Bearer " + k
	}
	resp, err := c.rawFetch(ctx, http.MethodPost, "/v1/chat/completions", params, headers)
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

// ChatCompletionsStream starts an OpenAI-compatible streaming completion and
// returns the raw SSE response. The caller can consume it with stream.ParseSSE.
func (c *Client) ChatCompletionsStream(ctx context.Context, params map[string]any, hp *HeadroomParams) (*http.Response, error) {
	params["stream"] = true
	headers := map[string]string{}
	if hp != nil && hp.Mode != "" {
		headers["x-headroom-mode"] = string(hp.Mode)
	}
	if k := c.ProviderAPIKey("OPENAI_API_KEY"); k != "" {
		headers["Authorization"] = "Bearer " + k
	}
	return c.rawFetch(ctx, http.MethodPost, "/v1/chat/completions", params, headers)
}

// ChatCompletionsSimulate posts to /v1/compress with simulate config and
// returns a SimulationResult.
func (c *Client) ChatCompletionsSimulate(ctx context.Context, model string, messages []Message) (*SimulationResult, error) {
	body := map[string]any{
		"messages": messages,
		"model":    model,
		"config": map[string]any{
			"default_mode":           "simulate",
			"generate_diff_artifact": true,
		},
	}
	raw, err := c.CompressRaw(ctx, body)
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}
	var out SimulationResult
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// MessagesCreate posts to /v1/messages (Anthropic-style passthrough).
func (c *Client) MessagesCreate(ctx context.Context, params map[string]any, hp *HeadroomParams) (map[string]any, error) {
	headers := map[string]string{
		"anthropic-version": "2023-06-01",
	}
	if hp != nil && hp.Mode != "" {
		headers["x-headroom-mode"] = string(hp.Mode)
	}
	if k := c.ProviderAPIKey("ANTHROPIC_API_KEY"); k != "" {
		headers["x-api-key"] = k
	}
	if _, ok := params["max_tokens"]; !ok {
		params["max_tokens"] = 1024
	}
	resp, err := c.rawFetch(ctx, http.MethodPost, "/v1/messages", params, headers)
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

// MessagesStream starts an Anthropic-compatible streaming message request and
// returns the raw SSE response. The caller can consume it with stream.ParseSSE.
func (c *Client) MessagesStream(ctx context.Context, params map[string]any, hp *HeadroomParams) (*http.Response, error) {
	params["stream"] = true
	if _, ok := params["max_tokens"]; !ok {
		params["max_tokens"] = 1024
	}
	headers := map[string]string{"anthropic-version": "2023-06-01"}
	if hp != nil && hp.Mode != "" {
		headers["x-headroom-mode"] = string(hp.Mode)
	}
	if k := c.ProviderAPIKey("ANTHROPIC_API_KEY"); k != "" {
		headers["x-api-key"] = k
	}
	return c.rawFetch(ctx, http.MethodPost, "/v1/messages", params, headers)
}

// MessagesSimulate is the Anthropic counterpart of ChatCompletionsSimulate.
func (c *Client) MessagesSimulate(ctx context.Context, model string, messages []Message) (*SimulationResult, error) {
	return c.ChatCompletionsSimulate(ctx, model, messages)
}
