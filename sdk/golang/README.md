# headroom-go

Go SDK for the [Headroom](https://headroom.ai) compression proxy. Mirrors the
TypeScript SDK (`headroom-ai`) — proxy-only client with adapters for OpenAI,
Anthropic, and Google Gemini message formats.

## Install

```bash
go get github.com/headroomlabs-ai/headroom/sdk/golang
```

Headroom compresses LLM context server-side. Start a proxy first:

```bash
pip install "headroom-ai[proxy]"
headroom proxy --port 8787
```

## Quick start

```go
import (
    "context"
    headroom "github.com/headroomlabs-ai/headroom/sdk/golang"
)

ctx := context.Background()
client := headroom.NewClient(headroom.WithBaseURL("http://localhost:8787"))

msgs := []headroom.Message{
    {Role: "system", Content: headroom.TextContent("You are a helpful assistant.")},
    {Role: "user", Content: headroom.TextContent("Summarize this conversation...")},
}

res, err := client.Compress(ctx, msgs, headroom.CompressOptions{Model: "gpt-4o"})
if err != nil { panic(err) }
fmt.Printf("Saved %d tokens (%.0f%%)\n", res.TokensSaved, (1-res.CompressionRatio)*100)
```

## Format-agnostic compress

```go
import "github.com/headroomlabs-ai/headroom/sdk/golang"

// Anthropic-shape input — auto-detected, converted to OpenAI for the proxy,
// converted back to Anthropic in the result.
msgs := []any{
    map[string]any{"role": "user", "content": "hi"},
    map[string]any{"role": "assistant", "content": []any{
        map[string]any{"type": "text", "text": "hello"},
    }},
}
res, err := headroom.Compress(ctx, msgs, headroom.CompressOptions{Model: "claude-sonnet-4-5-20250929"})
```

## Simulate (dry run)

```go
sim, err := headroom.Simulate(ctx, msgs, headroom.SimulateOptions{Model: "gpt-4o"})
fmt.Printf("Would save %d tokens (%s)\n", sim.TokensSaved, sim.EstimatedSavings)
```

## OpenAI / Anthropic-style passthrough

```go
out, err := client.ChatCompletionsCreate(ctx, map[string]any{
    "model":    "gpt-4o",
    "messages": msgs,
}, &headroom.HeadroomParams{Mode: headroom.ModeOptimize})
```

## Streaming

```go
import "github.com/headroomlabs-ai/headroom/sdk/golang/stream"

resp, _ := client.ChatCompletionsStream(ctx, map[string]any{
    "model": "gpt-4o", "messages": msgs,
}, nil)
for ev, err := range stream.ParseSSE(resp) {
    if err != nil { break }
    // handle ev (map[string]any)
}

// Channel fallback (for older Go or non-iter callers):
for e := range stream.ParseSSEChan(resp) { /* … */ }
```

## SharedContext (multi-agent compressed handoffs)

```go
sc := headroom.NewSharedContext(headroom.SharedContextOptions{Client: client})
entry, _ := sc.Put(ctx, "research", bigAgentOutput, "researcher")
summary, _ := sc.Get("research", false) // compressed
full, _ := sc.Get("research", true)     // original
```

## Adapters

```go
import "github.com/headroomlabs-ai/headroom/sdk/golang/openai"
import "github.com/headroomlabs-ai/headroom/sdk/golang/anthropic"
import "github.com/headroomlabs-ai/headroom/sdk/golang/gemini"

compressed, err := openai.WithHeadroom(ctx, msgs)
```

## Errors

```go
if err != nil {
    if errors.Is(err, headroom.ErrAuth) {
        // 401
    }
    if ce, ok := headroom.AsCompressError(err); ok {
        fmt.Println(ce.StatusCode, ce.ErrorType)
    }
}
```

## Environment

- `HEADROOM_BASE_URL` — proxy URL (default `http://localhost:8787`)
- `HEADROOM_API_KEY` — Cloud API key (only needed for Headroom Cloud)

## Testing

```bash
# Unit tests (no proxy needed)
go test ./...

# Integration tests against a running proxy
HEADROOM_TEST_BASE_URL=http://localhost:8787 go test ./...
```

## License

Apache-2.0
