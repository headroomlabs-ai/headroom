// Package stream parses Server-Sent Events from a *http.Response body.
//
// ParseSSE returns an iter.Seq2[map[string]any, error] (Go 1.23+, the modern
// pull iterator). For older Go or callers who prefer channels, ParseSSEChan
// returns the same events on a buffered channel.
package stream

import (
	"bufio"
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"strings"
)

// ParseSSE returns an iterator over SSE JSON events.
// The iterator yields (event, error). On error it stops; "[DONE]" sentinel
// also ends the stream cleanly. The caller is responsible for closing
// resp.Body when done (or letting the iterator drain it fully).
func ParseSSE(resp *http.Response) iter.Seq2[map[string]any, error] {
	return func(yield func(map[string]any, error) bool) {
		defer resp.Body.Close()
		br := bufio.NewReader(resp.Body)
		for {
			line, err := br.ReadString('\n')
			if len(line) > 0 {
				line = strings.TrimRight(line, "\r\n")
				if strings.HasPrefix(line, "data: ") {
					data := strings.TrimSpace(line[len("data: "):])
					if data == "[DONE]" {
						return
					}
					var ev map[string]any
					if jerr := json.Unmarshal([]byte(data), &ev); jerr == nil {
						if !yield(ev, nil) {
							return
						}
					}
					// Skip non-JSON data lines silently (matches TS behavior).
				}
			}
			if err != nil {
				if err != io.EOF {
					yield(nil, err)
				}
				return
			}
		}
	}
}

// ParseSSEChan is a chan-based fallback. Closes the channel when the stream
// ends. Caller must drain the channel or cancel via the response body close.
func ParseSSEChan(resp *http.Response) <-chan SSEEvent {
	ch := make(chan SSEEvent, 16)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		br := bufio.NewReader(resp.Body)
		for {
			line, err := br.ReadString('\n')
			if len(line) > 0 {
				line = strings.TrimRight(line, "\r\n")
				if strings.HasPrefix(line, "data: ") {
					data := strings.TrimSpace(line[len("data: "):])
					if data == "[DONE]" {
						return
					}
					var ev map[string]any
					if jerr := json.Unmarshal([]byte(data), &ev); jerr == nil {
						ch <- SSEEvent{Data: ev}
					}
				}
			}
			if err != nil {
				if err != io.EOF {
					ch <- SSEEvent{Err: err}
				}
				return
			}
		}
	}()
	return ch
}

// SSEEvent is one item from ParseSSEChan — either Data or Err is set.
type SSEEvent struct {
	Data map[string]any
	Err  error
}

// Collect drains an iter.Seq2 into a slice. Stops at first error.
func Collect(seq iter.Seq2[map[string]any, error]) ([]map[string]any, error) {
	var out []map[string]any
	for ev, err := range seq {
		if err != nil {
			return out, err
		}
		out = append(out, ev)
	}
	return out, nil
}
