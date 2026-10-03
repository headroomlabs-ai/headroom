// Handwritten JSON transport kernel, copied verbatim by sdkgen. No retries.
package headroomwire

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"
)

//go:embed schemas.json
var schemaBytes []byte

type schema struct {
	Ref        string            `json:"$ref"`
	Type       string            `json:"type"`
	AnyOf      []schema          `json:"anyOf"`
	Enum       []any             `json:"enum"`
	Properties map[string]schema `json:"properties"`
	Required   []string          `json:"required"`
	Additional json.RawMessage   `json:"additionalProperties"`
	Items      *schema           `json:"items"`
}
var schemas = func() map[string]schema {
	var out map[string]schema
	if err := json.Unmarshal(schemaBytes, &out); err != nil {
		panic("invalid generated schema")
	}
	return out
}()

// Optional distinguishes omitted, explicit null, and a supplied value.
// Only fields declared nullable may use Null; validators reject null elsewhere.
type Optional[T any] struct {
	Set   bool
	Value *T
}

func Some[T any](v T) Optional[T] { return Optional[T]{Set: true, Value: &v} }
func Null[T any]() Optional[T]    { return Optional[T]{Set: true} }
func (o *Optional[T]) UnmarshalJSON(raw []byte) error {
	o.Set = true
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		o.Value = nil
		return nil
	}
	var value T
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	o.Value = &value
	return nil
}
func (o Optional[T]) MarshalJSON() ([]byte, error) { return json.Marshal(o.Value) }

type APIError struct {
	Status  int
	Headers http.Header
	Body    []byte
}

func (e *APIError) Error() string { return fmt.Sprintf("Headroom HTTP %d", e.Status) }

type Options struct {
	Headers          http.Header
	HTTPClient       *http.Client
	Timeout          time.Duration
	MaxResponseBytes int64
}
type Transport struct {
	base     string
	headers  http.Header
	client   *http.Client
	maxBytes int64
}

func newTransport(base string, options *Options) (*Transport, error) {
	if base == "" {
		base = "http://localhost:8787"
	}
	parsed, err := url.Parse(base)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" || strings.ContainsAny(base, "\r\n\t ") {
		return nil, fmt.Errorf("invalid base URL")
	}
	opts := Options{}
	if options != nil {
		opts = *options
	}
	if opts.Timeout < 0 || opts.MaxResponseBytes < 0 {
		return nil, fmt.Errorf("invalid transport limits")
	}
	if opts.Timeout == 0 {
		opts.Timeout = 30 * time.Second
	}
	if opts.MaxResponseBytes == 0 {
		opts.MaxResponseBytes = 16 * 1024 * 1024
	}
	client := http.Client{Timeout: opts.Timeout}
	if opts.HTTPClient != nil {
		client = *opts.HTTPClient
		if client.Timeout == 0 {
			client.Timeout = opts.Timeout
		}
	}
	if client.Transport == nil {
		// Bypass environment proxies unless an explicit RoundTripper was supplied.
		tr := http.DefaultTransport.(*http.Transport).Clone()
		tr.Proxy = nil
		// net/http retries idempotent requests only after a reused connection
		// fails. Fresh connections make every SDK call a single attempt.
		tr.DisableKeepAlives = true
		client.Transport = tr
	}
	client.CheckRedirect = func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }
	return &Transport{base: strings.TrimRight(base, "/"), headers: opts.Headers.Clone(), client: &client, maxBytes: opts.MaxResponseBytes}, nil
}
func pathSegment(v string) (string, error) {
	if v == "" || v == "." || v == ".." {
		return "", fmt.Errorf("invalid path segment")
	}
	// QueryEscape escapes slash, plus, and other segment delimiters; use %20 rather than +.
	return strings.ReplaceAll(url.QueryEscape(v), "+", "%20"), nil
}
func (t *Transport) request(ctx context.Context, method, path string, body any, out any, requestModel string) error {
	var payload []byte
	var err error
	if requestModel != "" {
		payload, err = json.Marshal(body)
		if err != nil {
			return err
		}
		if err = validateModel(payload, requestModel); err != nil {
			return err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, t.base+path, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header = t.headers.Clone()
	if req.Header == nil {
		req.Header = make(http.Header)
	}
	req.Header.Set("Accept", "application/json")
	if requestModel != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, t.maxBytes+1))
	if err != nil {
		return err
	}
	if int64(len(raw)) > t.maxBytes {
		return fmt.Errorf("response exceeds configured byte limit")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &APIError{Status: resp.StatusCode, Headers: resp.Header.Clone(), Body: raw}
	}
	media := strings.ToLower(strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0]))
	if media != "application/json" && !strings.HasSuffix(media, "+json") {
		return fmt.Errorf("expected a JSON Content-Type")
	}
	return json.Unmarshal(raw, out) // Generated models validate and preserve unknown fields.
}
func validateModel(raw []byte, model string) error {
	if !utf8.Valid(raw) {
		return fmt.Errorf("invalid UTF-8 JSON")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var value any
	if err := dec.Decode(&value); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("trailing JSON content")
	}
	s, ok := schemas[model]
	if !ok {
		return fmt.Errorf("unknown generated model")
	}
	return validateValue(value, s, "$", 0)
}
func validateValue(v any, s schema, path string, depth int) error {
	if depth > 100 {
		return fmt.Errorf("%s: maximum JSON nesting exceeded", path)
	}
	if s.Ref != "" {
		parts := strings.Split(s.Ref, "/")
		return validateValue(v, schemas[parts[len(parts)-1]], path, depth+1)
	}
	if len(s.AnyOf) > 0 {
		for _, part := range s.AnyOf {
			if validateValue(v, part, path, depth+1) == nil {
				return nil
			}
		}
		return fmt.Errorf("%s: outside nullable union", path)
	}
	good := true
	switch s.Type {
	case "string":
		_, good = v.(string)
	case "integer":
		n, ok := v.(json.Number)
		good = ok
		if ok {
			_, err := n.Int64()
			good = err == nil
		}
	case "number":
		n, ok := v.(json.Number)
		good = ok
		if ok {
			f, err := n.Float64()
			good = err == nil && !math.IsInf(f, 0) && !math.IsNaN(f)
		}
	case "boolean":
		_, good = v.(bool)
	case "null":
		good = v == nil
	case "array":
		_, good = v.([]any)
	case "object":
		_, good = v.(map[string]any)
	}
	if !good {
		return fmt.Errorf("%s: expected %s", path, s.Type)
	}
	if len(s.Enum) > 0 {
		found := false
		raw, _ := json.Marshal(v)
		for _, item := range s.Enum {
			candidate, _ := json.Marshal(item)
			if bytes.Equal(raw, candidate) {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("%s: unknown enum value", path)
		}
	}
	if object, ok := v.(map[string]any); ok {
		for _, key := range s.Required {
			if _, ok := object[key]; !ok {
				return fmt.Errorf("%s.%s: required field missing", path, key)
			}
		}
		for key, value := range object {
			rule, found := s.Properties[key]
			if !found && len(s.Additional) > 0 {
				if string(s.Additional) == "false" {
					return fmt.Errorf("%s.%s: extra field forbidden", path, key)
				}
				if string(s.Additional) != "true" {
					if err := json.Unmarshal(s.Additional, &rule); err != nil {
						return err
					}
				}
			}
			if err := validateValue(value, rule, path+"."+key, depth+1); err != nil {
				return err
			}
		}
	}
	if list, ok := v.([]any); ok {
		rule := schema{}
		if s.Items != nil {
			rule = *s.Items
		}
		for _, value := range list {
			if err := validateValue(value, rule, path+"[]", depth+1); err != nil {
				return err
			}
		}
	}
	return nil
}
