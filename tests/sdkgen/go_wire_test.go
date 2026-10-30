package headroomwire

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

func fixtureClient(t *testing.T, options *Options) *Client {
	t.Helper()
	c, err := NewClient(os.Getenv("HEADROOM_WIRE_TEST_URL"), options)
	if err != nil {
		t.Fatal(err)
	}
	return c
}
func TestPostNullUnicodeAndFutureFields(t *testing.T) {
	c := fixtureClient(t, nil)
	out, err := c.Retrieve(context.Background(), RetrieveRequest{Hash: "ok", AdditionalProperties: map[string]json.RawMessage{"extra_request": json.RawMessage(`{"snake_case":false}`)}})
	if err != nil {
		t.Fatal(err)
	}
	if out.ToolName != nil || out.OriginalContent != `{"snake_case":"世界"}` {
		t.Fatalf("wire mismatch: %#v", out)
	}
	if string(out.AdditionalProperties["future_extension"]) != `{"snake_case":"unchanged"}` {
		t.Fatal("future field lost")
	}
	raw, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"tool_name":null`) || !strings.Contains(string(raw), `"future_extension"`) {
		t.Fatal("roundtrip lost data")
	}
}
func TestPathEncoding(t *testing.T) {
	value := "a/世界 ?#+'!*()"
	c := fixtureClient(t, nil)
	out, err := c.RetrieveGet(context.Background(), value)
	if err != nil {
		t.Fatal(err)
	}
	if out.Hash != value {
		t.Fatal("path changed")
	}
}
func TestHTTPError(t *testing.T) {
	_, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: "missing"})
	var api *APIError
	if !errors.As(err, &api) || api.Status != 404 || api.Headers.Get("X-Fixture") != "true" {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(string(api.Body), "Entry missing") || strings.Contains(api.Error(), "Entry missing") {
		t.Fatal("error payload handling")
	}
}
func TestMissingRequiredNullable(t *testing.T) {
	_, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: "missing_field"})
	if err == nil {
		t.Fatal("missing required field accepted")
	}
}
func TestWrongType(t *testing.T) {
	_, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: "wrong_type"})
	if err == nil {
		t.Fatal("wrong type accepted")
	}
}
func TestNullNonNullableAndValidNonJSON(t *testing.T) {
	for _, key := range []string{"null_nonnullable", "valid_nonjson"} {
		if _, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: key}); err == nil {
			t.Fatalf("invalid response %s accepted", key)
		}
	}
}
func TestNoAutomaticRetry(t *testing.T) {
	if _, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: "retry_probe_go"}); err == nil {
		t.Fatal("closed connection accepted")
	}
}
func TestWideIntegerIsExact(t *testing.T) {
	out, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: "unsafe_int"})
	if err != nil {
		t.Fatal(err)
	}
	if out.OriginalTokens != 9007199254740993 {
		t.Fatal("integer rounded")
	}
}
func TestNonJSON(t *testing.T) {
	_, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: "notjson"})
	if err == nil {
		t.Fatal("non-JSON accepted")
	}
}
func TestNoRedirect(t *testing.T) {
	_, err := fixtureClient(t, nil).Retrieve(context.Background(), RetrieveRequest{Hash: "redirect"})
	var api *APIError
	if !errors.As(err, &api) || api.Status != 302 {
		t.Fatalf("redirect followed: %v", err)
	}
}
func TestBoundedRead(t *testing.T) {
	_, err := fixtureClient(t, &Options{MaxResponseBytes: 16}).Retrieve(context.Background(), RetrieveRequest{Hash: "ok"})
	if err == nil {
		t.Fatal("unbounded response")
	}
}
func TestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := fixtureClient(t, nil).Retrieve(ctx, RetrieveRequest{Hash: "ok"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}
func TestTimeoutAndInFlightCancellation(t *testing.T) {
	if _, err := fixtureClient(t, &Options{Timeout: 50 * time.Millisecond}).Retrieve(context.Background(), RetrieveRequest{Hash: "slow"}); err == nil {
		t.Fatal("timeout accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(25*time.Millisecond, cancel)
	if _, err := fixtureClient(t, nil).Retrieve(ctx, RetrieveRequest{Hash: "slow"}); err == nil {
		t.Fatal("in-flight cancellation accepted")
	}
}

func TestRequestTimeoutWithCustomHTTPClient(t *testing.T) {
	custom := &http.Client{Timeout: 30 * time.Second}
	_, err := fixtureClient(t, &Options{Timeout: 50 * time.Millisecond, HTTPClient: custom}).Retrieve(context.Background(), RetrieveRequest{Hash: "slow"})
	if err == nil {
		t.Fatal("custom HTTP client overrode request timeout")
	}
	if custom.Timeout != 30*time.Second {
		t.Fatal("caller HTTP client was mutated")
	}
}
func TestInvalidURLAndPath(t *testing.T) {
	for _, base := range []string{"file:///tmp", "http://user:pass@localhost", "http://localhost/?secret=1"} {
		if _, err := NewClient(base, nil); err == nil {
			t.Fatal("invalid base accepted")
		}
	}
	if _, err := fixtureClient(t, nil).RetrieveGet(context.Background(), ".."); err == nil {
		t.Fatal("dot segment accepted")
	}
}
func TestAdditionalPropertyCollision(t *testing.T) {
	_, err := json.Marshal(RetrieveRequest{Hash: "ok", AdditionalProperties: map[string]json.RawMessage{"hash": json.RawMessage(`"other"`)}})
	if err == nil {
		t.Fatal("declared field overwritten")
	}
}
