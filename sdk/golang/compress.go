package headroom

import (
	"context"

	"github.com/headroomlabs-ai/headroom/sdk/golang/format"
)

// Compress is the standalone, format-agnostic compress entry point.
// It accepts messages in any supported format ([]Message, []map[string]any
// in Anthropic/Vercel/Gemini shape, etc.) as []any, detects the format,
// converts to OpenAI for the proxy, and converts the result back.
func Compress(ctx context.Context, messages []any, opts CompressOptions) (*CompressResultGeneric, error) {
	model := opts.Model
	if model == "" {
		model = "gpt-4o"
	}

	hookCtx := CompressContext{
		Model:      model,
		UserQuery:  ExtractUserQuery(messages),
		TurnNumber: CountTurns(messages),
		ToolCalls:  ExtractToolCalls(messages),
	}

	processed := messages
	if opts.Hooks != nil {
		out, err := opts.Hooks.PreCompress(ctx, messages, hookCtx)
		if err != nil {
			return nil, err
		}
		processed = out
	}

	inputFormat := format.Detect(processed)
	openaiAny := format.ToOpenAI(processed)
	openaiMsgs, err := anyToMessages(openaiAny)
	if err != nil {
		return nil, err
	}

	if opts.Hooks != nil {
		// Biases are computed but not yet wired to the proxy in TS either —
		// kept here for interface parity.
		_, _ = opts.Hooks.ComputeBiases(ctx, openaiMsgs, hookCtx)
	}

	client := opts.Client
	if client == nil {
		client = newClientFromOptions(ClientOptions{
			BaseURL:  opts.BaseURL,
			APIKey:   opts.APIKey,
			Timeout:  opts.Timeout,
			Retries:  opts.Retries,
			Fallback: opts.Fallback,
		})
	}

	res, err := client.Compress(ctx, openaiMsgs, CompressOptions{Model: model, TokenBudget: opts.TokenBudget})
	if err != nil {
		return nil, err
	}

	outAny := format.FromOpenAI(messagesAsAny(res.Messages), inputFormat)

	out := &CompressResultGeneric{
		Messages:          outAny,
		TokensBefore:      res.TokensBefore,
		TokensAfter:       res.TokensAfter,
		TokensSaved:       res.TokensSaved,
		CompressionRatio:  res.CompressionRatio,
		TransformsApplied: res.TransformsApplied,
		CCRHashes:         res.CCRHashes,
		Compressed:        res.Compressed,
	}

	if opts.Hooks != nil {
		_ = opts.Hooks.PostCompress(ctx, CompressEvent{
			TokensBefore:      res.TokensBefore,
			TokensAfter:       res.TokensAfter,
			TokensSaved:       res.TokensSaved,
			CompressionRatio:  res.CompressionRatio,
			TransformsApplied: res.TransformsApplied,
			CCRHashes:         res.CCRHashes,
			Model:             hookCtx.Model,
			UserQuery:         hookCtx.UserQuery,
			Provider:          hookCtx.Provider,
		})
	}

	return out, nil
}
