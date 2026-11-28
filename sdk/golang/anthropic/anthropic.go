// Package anthropic is the Anthropic-flavored adapter. Pass Anthropic-shape
// messages (content blocks, tool_use, tool_result); WithHeadroom converts
// them to OpenAI for the proxy and converts the compressed output back.
package anthropic

import (
	"context"

	headroom "github.com/headroomlabs-ai/headroom/sdk/golang"
)

// Compress takes Anthropic-shape messages ([]any with content-block arrays),
// runs them through the proxy, and returns Anthropic-shape compressed output.
func Compress(ctx context.Context, messages []any, opts headroom.CompressOptions) (*headroom.CompressResultGeneric, error) {
	return headroom.Compress(ctx, messages, opts)
}

// WithHeadroom returns the compressed messages directly (no metadata).
func WithHeadroom(ctx context.Context, messages []any, opts ...headroom.Option) ([]any, error) {
	o := headroom.ClientOptions{}
	for _, fn := range opts {
		fn(&o)
	}
	res, err := headroom.Compress(ctx, messages, headroom.CompressOptions{
		BaseURL:  o.BaseURL,
		APIKey:   o.APIKey,
		Timeout:  o.Timeout,
		Retries:  o.Retries,
		Fallback: o.Fallback,
	})
	if err != nil {
		return nil, err
	}
	return res.Messages, nil
}
