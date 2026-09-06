// Package openai is the OpenAI-flavored adapter — call WithHeadroom on
// any messages slice or use Compress directly. No external SDK dependency:
// users keep their own OpenAI client and we just compress messages first.
package openai

import (
	"context"

	headroom "github.com/headroomlabs-ai/headroom/sdk/golang"
)

// Compress shrinks an OpenAI []Message slice via the proxy. If the proxy
// is unreachable and Fallback is enabled (default), the original messages
// are returned untouched.
func Compress(ctx context.Context, messages []headroom.Message, opts headroom.CompressOptions) (*headroom.CompressResult, error) {
	client := opts.Client
	if client == nil {
		client = headroom.NewClient(
			headroom.WithBaseURL(opts.BaseURL),
			headroom.WithAPIKey(opts.APIKey),
			headroom.WithRetries(opts.Retries),
		)
	}
	return client.Compress(ctx, messages, opts)
}

// WithHeadroom is the convenience wrapper used in docs:
//
//	compressed, err := openai.WithHeadroom(ctx, messages)
//
// It uses default options (proxy at HEADROOM_BASE_URL or localhost:8787).
func WithHeadroom(ctx context.Context, messages []headroom.Message, opts ...headroom.Option) ([]headroom.Message, error) {
	client := headroom.NewClient(opts...)
	res, err := client.Compress(ctx, messages, headroom.CompressOptions{})
	if err != nil {
		return nil, err
	}
	return res.Messages, nil
}
