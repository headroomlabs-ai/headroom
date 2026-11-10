package headroom

import (
	"errors"
	"testing"
)

func TestMapProxyError(t *testing.T) {
	cases := []struct {
		status   int
		errType  string
		wantKind string
	}{
		{401, "ignored", "auth"},
		{500, "configuration_error", "configuration"},
		{500, "provider_error", "provider"},
		{500, "storage_error", "storage"},
		{500, "tokenization_error", "tokenization"},
		{500, "cache_error", "cache"},
		{500, "validation_error", "validation"},
		{500, "transform_error", "transform"},
		{500, "unknown_thing", "compress"},
	}
	for _, c := range cases {
		err := MapProxyError(c.status, c.errType, "msg")
		var target *Error
		switch c.wantKind {
		case "auth":
			target = ErrAuth
		case "configuration":
			target = ErrConfiguration
		case "provider":
			target = ErrProvider
		case "storage":
			target = ErrStorage
		case "tokenization":
			target = ErrTokenization
		case "cache":
			target = ErrCache
		case "validation":
			target = ErrValidation
		case "transform":
			target = ErrTransform
		case "compress":
			target = ErrCompress
		}
		if !errors.Is(err, target) {
			t.Errorf("status=%d type=%s: want kind %s, got %#v", c.status, c.errType, c.wantKind, err)
		}
	}
}

func TestCompressErrorCarriesStatus(t *testing.T) {
	err := MapProxyError(503, "unknown", "boom")
	ce, ok := AsCompressError(err)
	if !ok {
		t.Fatalf("expected CompressError, got %T", err)
	}
	if ce.StatusCode != 503 {
		t.Errorf("status: want 503, got %d", ce.StatusCode)
	}
	if ce.ErrorType != "unknown" {
		t.Errorf("type: want unknown, got %s", ce.ErrorType)
	}
}
