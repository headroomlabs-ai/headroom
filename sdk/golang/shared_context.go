package headroom

import (
	"context"
	"encoding/json"
	"sync"
	"time"
)

// ContextEntry is a single SharedContext slot.
type ContextEntry struct {
	Key              string   `json:"key"`
	Original         string   `json:"original"`
	Compressed       string   `json:"compressed"`
	OriginalTokens   int      `json:"original_tokens"`
	CompressedTokens int      `json:"compressed_tokens"`
	Agent            string   `json:"agent,omitempty"`
	Timestamp        float64  `json:"timestamp"`
	Transforms       []string `json:"transforms"`
	SavingsPercent   float64  `json:"savings_percent"`
}

// SharedContextStats summarizes a SharedContext.
type SharedContextStats struct {
	Entries               int     `json:"entries"`
	TotalOriginalTokens   int     `json:"total_original_tokens"`
	TotalCompressedTokens int     `json:"total_compressed_tokens"`
	TotalTokensSaved      int     `json:"total_tokens_saved"`
	SavingsPercent        float64 `json:"savings_percent"`
}

// SharedContextOptions configures a SharedContext.
type SharedContextOptions struct {
	BaseURL    string
	APIKey     string
	Timeout    time.Duration
	Model      string
	TTL        time.Duration // 0 -> default 1h
	MaxEntries int
	Client     *Client
}

// SharedContext is a compressed inter-agent K/V store.
// Thread-safe.
type SharedContext struct {
	mu         sync.Mutex
	entries    map[string]*ContextEntry
	order      []string // FIFO for eviction
	client     *Client
	model      string
	ttl        time.Duration
	maxEntries int
}

// NewSharedContext builds a SharedContext.
func NewSharedContext(opts SharedContextOptions) *SharedContext {
	model := opts.Model
	if model == "" {
		model = "claude-sonnet-4-5-20250929"
	}
	ttl := opts.TTL
	if ttl == 0 {
		ttl = time.Hour
	}
	max := opts.MaxEntries
	if max == 0 {
		max = 100
	}
	client := opts.Client
	if client == nil {
		client = newClientFromOptions(ClientOptions{BaseURL: opts.BaseURL, APIKey: opts.APIKey, Timeout: opts.Timeout})
	}
	return &SharedContext{
		entries:    map[string]*ContextEntry{},
		client:     client,
		model:      model,
		ttl:        ttl,
		maxEntries: max,
	}
}

// Put compresses content and stores it under key.
func (s *SharedContext) Put(ctx context.Context, key, content, agent string) (*ContextEntry, error) {
	s.mu.Lock()
	s.evictExpiredLocked()
	s.evictIfFullLocked()
	s.mu.Unlock()

	msg := Message{Role: "user", Content: TextContent(content)}
	res, err := s.client.Compress(ctx, []Message{msg}, CompressOptions{Model: s.model})
	if err != nil || res == nil {
		// Fall back to uncompressed storage.
		res = &CompressResult{
			Messages:         []Message{msg},
			TokensBefore:     len(content) / 4,
			TokensAfter:      len(content) / 4,
			CompressionRatio: 1.0,
			Compressed:       false,
		}
	}

	compressed := content
	if res.Compressed && len(res.Messages) > 0 {
		var c any
		_ = json.Unmarshal(res.Messages[0].Content, &c)
		switch v := c.(type) {
		case string:
			compressed = v
		default:
			b, _ := json.Marshal(v)
			compressed = string(b)
		}
	}

	entry := &ContextEntry{
		Key:              key,
		Original:         content,
		Compressed:       compressed,
		OriginalTokens:   res.TokensBefore,
		CompressedTokens: res.TokensAfter,
		Agent:            agent,
		Timestamp:        float64(time.Now().UnixNano()) / 1e9,
		Transforms:       res.TransformsApplied,
	}
	if res.TokensBefore > 0 {
		entry.SavingsPercent = float64(res.TokensBefore-res.TokensAfter) / float64(res.TokensBefore) * 100
	}

	s.mu.Lock()
	if _, exists := s.entries[key]; !exists {
		s.order = append(s.order, key)
	}
	s.entries[key] = entry
	s.mu.Unlock()
	return entry, nil
}

// Get returns the compressed payload (or original if full=true). Returns "" and false if missing/expired.
func (s *SharedContext) Get(key string, full bool) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[key]
	if !ok {
		return "", false
	}
	if s.expiredLocked(e) {
		s.deleteLocked(key)
		return "", false
	}
	if full {
		return e.Original, true
	}
	return e.Compressed, true
}

// GetEntry returns the full entry; nil if missing/expired.
func (s *SharedContext) GetEntry(key string) *ContextEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[key]
	if !ok {
		return nil
	}
	if s.expiredLocked(e) {
		s.deleteLocked(key)
		return nil
	}
	return e
}

// Keys returns non-expired keys.
func (s *SharedContext) Keys() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictExpiredLocked()
	out := make([]string, 0, len(s.entries))
	for _, k := range s.order {
		if _, ok := s.entries[k]; ok {
			out = append(out, k)
		}
	}
	return out
}

// Stats returns aggregated metrics.
func (s *SharedContext) Stats() SharedContextStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictExpiredLocked()
	stat := SharedContextStats{Entries: len(s.entries)}
	for _, e := range s.entries {
		stat.TotalOriginalTokens += e.OriginalTokens
		stat.TotalCompressedTokens += e.CompressedTokens
	}
	stat.TotalTokensSaved = stat.TotalOriginalTokens - stat.TotalCompressedTokens
	if stat.TotalOriginalTokens > 0 {
		stat.SavingsPercent = float64(stat.TotalTokensSaved) / float64(stat.TotalOriginalTokens) * 100
	}
	return stat
}

// Clear removes all entries.
func (s *SharedContext) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = map[string]*ContextEntry{}
	s.order = nil
}

func (s *SharedContext) expiredLocked(e *ContextEntry) bool {
	now := float64(time.Now().UnixNano()) / 1e9
	return now-e.Timestamp > s.ttl.Seconds()
}

func (s *SharedContext) evictExpiredLocked() {
	now := float64(time.Now().UnixNano()) / 1e9
	for k, e := range s.entries {
		if now-e.Timestamp > s.ttl.Seconds() {
			s.deleteLocked(k)
		}
	}
}

func (s *SharedContext) evictIfFullLocked() {
	for len(s.entries) >= s.maxEntries && len(s.order) > 0 {
		oldest := s.order[0]
		s.deleteLocked(oldest)
	}
}

func (s *SharedContext) deleteLocked(key string) {
	delete(s.entries, key)
	for i, k := range s.order {
		if k == key {
			s.order = append(s.order[:i], s.order[i+1:]...)
			return
		}
	}
}
