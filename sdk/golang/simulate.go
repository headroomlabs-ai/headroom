package headroom

import (
	"context"
	"encoding/json"

	"github.com/headroomlabs-ai/headroom/sdk/golang/format"
)

// SimulateOptions controls a simulation call.
type SimulateOptions struct {
	Model   string
	BaseURL string
	APIKey  string
	Config  *Config
	Client  *Client
}

// Simulate dry-runs compression: no LLM call, but the proxy returns what it
// *would* compress. Pass any-format messages; they're auto-converted to OpenAI.
func Simulate(ctx context.Context, messages []any, opts SimulateOptions) (*SimulationResult, error) {
	model := opts.Model
	if model == "" {
		model = "gpt-4o"
	}

	openaiAny := format.ToOpenAI(messages)
	openaiMsgs, err := anyToMessages(openaiAny)
	if err != nil {
		return nil, err
	}

	cfg := map[string]any{
		"default_mode":           "simulate",
		"generate_diff_artifact": true,
	}
	if opts.Config != nil {
		b, _ := json.Marshal(opts.Config)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		for k, v := range m {
			cfg[k] = v
		}
		// Force simulate mode regardless of provided config.
		cfg["default_mode"] = "simulate"
		cfg["generate_diff_artifact"] = true
	}

	client := opts.Client
	if client == nil {
		client = newClientFromOptions(ClientOptions{BaseURL: opts.BaseURL, APIKey: opts.APIKey})
	}

	body := map[string]any{
		"messages": openaiMsgs,
		"model":    model,
		"config":   cfg,
	}
	raw, err := client.CompressRaw(ctx, body)
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
