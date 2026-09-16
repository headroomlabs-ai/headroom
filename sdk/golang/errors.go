package headroom

import (
	"errors"
	"fmt"
)

// Error is the base type for every Headroom SDK error.
// Use errors.As to inspect specific subtypes (ConnectionError, AuthError, …).
type Error struct {
	Kind    string // e.g. "connection", "auth", "compress", "configuration"
	Message string
	Details map[string]any
}

func (e *Error) Error() string { return e.Message }

// Sentinels for errors.Is matching on Kind.
var (
	ErrConnection    = &Error{Kind: "connection", Message: "connection error"}
	ErrAuth          = &Error{Kind: "auth", Message: "auth error"}
	ErrCompress      = &Error{Kind: "compress", Message: "compress error"}
	ErrConfiguration = &Error{Kind: "configuration", Message: "configuration error"}
	ErrProvider      = &Error{Kind: "provider", Message: "provider error"}
	ErrStorage       = &Error{Kind: "storage", Message: "storage error"}
	ErrTokenization  = &Error{Kind: "tokenization", Message: "tokenization error"}
	ErrCache         = &Error{Kind: "cache", Message: "cache error"}
	ErrValidation    = &Error{Kind: "validation", Message: "validation error"}
	ErrTransform     = &Error{Kind: "transform", Message: "transform error"}
)

// Is implements errors.Is by Kind so callers can write
//
//	if errors.Is(err, headroom.ErrAuth) { … }
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	if !ok {
		return false
	}
	return e.Kind == t.Kind
}

// CompressError carries the proxy's HTTP status and error type.
// It is also a *Error (kind="compress").
type CompressError struct {
	Err        *Error
	StatusCode int
	ErrorType  string
}

func (e *CompressError) Error() string { return e.Err.Error() }
func (e *CompressError) Unwrap() error { return e.Err }

func newError(kind, message string, details map[string]any) *Error {
	return &Error{Kind: kind, Message: message, Details: details}
}

// MapProxyError maps an HTTP status + proxy error type to a typed SDK error.
func MapProxyError(status int, errType, message string) error {
	if status == 401 {
		return newError("auth", message, nil)
	}
	switch errType {
	case "configuration_error":
		return newError("configuration", message, map[string]any{"status_code": status, "error_type": errType})
	case "provider_error":
		return newError("provider", message, map[string]any{"status_code": status, "error_type": errType})
	case "storage_error":
		return newError("storage", message, map[string]any{"status_code": status, "error_type": errType})
	case "tokenization_error":
		return newError("tokenization", message, map[string]any{"status_code": status, "error_type": errType})
	case "cache_error":
		return newError("cache", message, map[string]any{"status_code": status, "error_type": errType})
	case "validation_error":
		return newError("validation", message, map[string]any{"status_code": status, "error_type": errType})
	case "transform_error":
		return newError("transform", message, map[string]any{"status_code": status, "error_type": errType})
	}
	return &CompressError{
		Err:        &Error{Kind: "compress", Message: message},
		StatusCode: status,
		ErrorType:  errType,
	}
}

// AsCompressError unwraps a *CompressError if present.
func AsCompressError(err error) (*CompressError, bool) {
	var ce *CompressError
	if errors.As(err, &ce) {
		return ce, true
	}
	return nil, false
}

func wrapConnection(format string, args ...any) error {
	return newError("connection", fmt.Sprintf(format, args...), nil)
}
