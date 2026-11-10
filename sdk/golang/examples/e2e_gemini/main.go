// End-to-end: compress with headroom, send to real Gemini API, verify response.
// Run with:  GEMINI_API_KEY=... go run ./examples/e2e_gemini
//
// Mirrors test/e2e-gemini.mjs in the TS SDK so we can show parity.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	headroom "github.com/headroomlabs-ai/headroom/sdk/golang"
	"github.com/headroomlabs-ai/headroom/sdk/golang/gemini"
)

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "missing GEMINI_API_KEY in env")
		os.Exit(2)
	}
	baseURL := os.Getenv("HEADROOM_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8788"
	}

	// Same input shape as TS e2e: 100-item search results + conversation.
	items := make([]map[string]any, 100)
	for i := range items {
		items[i] = map[string]any{
			"id":      fmt.Sprintf("doc_%d", i),
			"title":   fmt.Sprintf("Search result number %d from the documentation index", i),
			"snippet": fmt.Sprintf("This is a moderately long snippet for result %d explaining what the document covers in some detail.", i),
			"score":   1.0 - float64(i)/100.0,
			"url":     fmt.Sprintf("https://example.com/docs/%d", i),
		}
	}
	itemsJSON, _ := json.Marshal(items)

	messages := []any{
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "You are a helpful assistant. Be concise."}}},
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "I just searched the docs. Here are the results — what are the top 3 by relevance?"}}},
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "Search results JSON:\n" + string(itemsJSON)}}},
	}

	fmt.Printf("headroom proxy: %s\n", baseURL)
	fmt.Printf("input: %d gemini messages, %d search items\n", len(messages), len(items))

	ctx := context.Background()
	client := headroom.NewClient(headroom.WithBaseURL(baseURL), headroom.WithTimeout(30*time.Second))

	// Step 1: compress through headroom (Gemini-shape in, Gemini-shape out).
	t0 := time.Now()
	res, err := gemini.Compress(ctx, messages, headroom.CompressOptions{
		Model:  "gemini-2.5-flash",
		Client: client,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "compress: %v\n", err)
		os.Exit(1)
	}
	t1 := time.Since(t0)

	fmt.Println("\n--- compression ---")
	fmt.Printf("tokens before: %d\n", res.TokensBefore)
	fmt.Printf("tokens after:  %d\n", res.TokensAfter)
	fmt.Printf("tokens saved:  %d\n", res.TokensSaved)
	fmt.Printf("transforms:    %v\n", res.TransformsApplied)
	shapeOK := true
	for _, m := range res.Messages {
		mm, ok := m.(map[string]any)
		if !ok {
			shapeOK = false
			break
		}
		if _, hasParts := mm["parts"]; !hasParts {
			shapeOK = false
			break
		}
	}
	fmt.Printf("compressed shape preserved: %v\n", shapeOK)
	fmt.Printf("compress latency: %v\n", t1)

	// Step 2: send the compressed Gemini-shape messages to gemini-2.5-flash.
	body, _ := json.Marshal(map[string]any{"contents": res.Messages})
	apiURL := "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", apiKey)

	t2 := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gemini call: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	t3 := time.Since(t2)

	if resp.StatusCode >= 400 {
		fmt.Fprintf(os.Stderr, "\nGemini API error %d: %s\n", resp.StatusCode, string(respBody[:min(500, len(respBody))]))
		os.Exit(1)
	}

	var data struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(respBody, &data); err != nil {
		fmt.Fprintf(os.Stderr, "decode: %v\nbody: %s\n", err, string(respBody[:min(500, len(respBody))]))
		os.Exit(1)
	}

	text := ""
	if len(data.Candidates) > 0 {
		for _, p := range data.Candidates[0].Content.Parts {
			text += p.Text
		}
	}

	fmt.Println("\n--- gemini response ---")
	fmt.Printf("api latency: %v\n", t3)
	fmt.Printf("response length: %d chars\n", len(text))
	fmt.Println("response preview:")
	fmt.Println(text[:min(600, len(text))])

	if len(text) < 10 {
		fmt.Fprintln(os.Stderr, "\nFAIL: response too short")
		os.Exit(1)
	}
	fmt.Println("\nPASS: Go SDK e2e (headroom compress + gemini-2.5-flash)")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
