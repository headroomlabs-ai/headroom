package headroom

import (
	"net/http"
	"time"
)

// ClientOptions configures a Client. Use the With* functions or build the
// struct directly. All fields are optional; defaults match the TS SDK.
type ClientOptions struct {
	BaseURL        string
	APIKey         string
	ProviderAPIKey string
	Timeout        time.Duration
	Retries        int
	Fallback       *bool // nil = default true
	HTTPClient     *http.Client
	Config         *Config
	DefaultMode    HeadroomMode
}

// Option is a functional option applied to ClientOptions.
type Option func(*ClientOptions)

func WithBaseURL(u string) Option           { return func(o *ClientOptions) { o.BaseURL = u } }
func WithAPIKey(k string) Option            { return func(o *ClientOptions) { o.APIKey = k } }
func WithProviderAPIKey(k string) Option    { return func(o *ClientOptions) { o.ProviderAPIKey = k } }
func WithTimeout(d time.Duration) Option    { return func(o *ClientOptions) { o.Timeout = d } }
func WithRetries(n int) Option              { return func(o *ClientOptions) { o.Retries = n } }
func WithFallback(b bool) Option            { return func(o *ClientOptions) { v := b; o.Fallback = &v } }
func WithHTTPClient(c *http.Client) Option  { return func(o *ClientOptions) { o.HTTPClient = c } }
func WithConfig(c *Config) Option           { return func(o *ClientOptions) { o.Config = c } }
func WithDefaultMode(m HeadroomMode) Option { return func(o *ClientOptions) { o.DefaultMode = m } }

// CompressOptions controls a single compress call.
type CompressOptions struct {
	Model       string
	TokenBudget int
	// ClientOptions used when no explicit Client is provided to Compress().
	BaseURL  string
	APIKey   string
	Timeout  time.Duration
	Retries  int
	Fallback *bool
	Hooks    Hooks
	Client   *Client
}

// HeadroomParams are extra request-scoped headroom_* options for chat.completions / messages.
type HeadroomParams struct {
	Mode               HeadroomMode
	CachePrefixTokens  int
	OutputBufferTokens int
	KeepTurns          int
	ToolProfiles       map[string]map[string]any
}
