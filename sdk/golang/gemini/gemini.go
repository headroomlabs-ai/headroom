// Package gemini is the Google Gemini adapter. Accepts {role, parts} shape
// messages, converts to OpenAI for compression, and converts back.
package gemini

import (
	"context"

	headroom "github.com/headroomlabs-ai/headroom/sdk/golang"
)

// Compress takes Gemini-shape messages and returns Gemini-shape compressed output.
func Compress(ctx context.Context, messages []any, opts headroom.CompressOptions) (*headroom.CompressResultGeneric, error) {
	return headroom.Compress(ctx, messages, opts)
}

// WithHeadroom returns the compressed messages directly.
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
