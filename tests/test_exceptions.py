from __future__ import annotations

from headroom.exceptions import (
    CacheError,
    CompressionError,
    ConfigurationError,
    HeadroomError,
    ProviderError,
    StorageError,
    TokenizationError,
    TransformError,
    ValidationError,
)


def test_headroom_error_formats_details() -> None:
    err = HeadroomError("bad config", details={"mode": "foo", "valid": "bar"})
    assert err.message == "bad config"
    assert err.details == {"mode": "foo", "valid": "bar"}
    assert str(err) == "bad config (mode=foo, valid=bar)"

    plain = HeadroomError("just bad")
    assert plain.details == {}
    assert str(plain) == "just bad"


def test_specialized_exceptions_inherit_headroom_error() -> None:
    for exc_type in (
        ConfigurationError,
        ProviderError,
        StorageError,
        CompressionError,
        TokenizationError,
        CacheError,
        ValidationError,
        TransformError,
    ):
        err = exc_type("problem", details={"kind": exc_type.__name__})
        assert isinstance(err, HeadroomError)
        assert str(err) == f"problem (kind={exc_type.__name__})"
