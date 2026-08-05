from __future__ import annotations

from fastapi import Request

from headroom.proxy.request_scope import normalize_request_path, normalize_scope_path


def test_normalize_scope_path_updates_path_and_raw_path() -> None:
    scope = {"path": "/old", "raw_path": b"/old"}

    normalize_scope_path(scope, "/v1/messages")

    assert scope == {"path": "/v1/messages", "raw_path": b"/v1/messages"}


def test_normalize_scope_path_quotes_raw_path_when_needed() -> None:
    scope = {"path": "/old", "raw_path": b"/old"}

    normalize_scope_path(scope, "/v1/models/claude opus")

    assert scope["path"] == "/v1/models/claude opus"
    assert scope["raw_path"] == b"/v1/models/claude%20opus"


def test_normalize_scope_path_leaves_missing_raw_path_absent() -> None:
    scope = {"path": "/old"}

    normalize_scope_path(scope, "/new")

    assert scope == {"path": "/new"}


def test_normalize_request_path_clears_cached_url() -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/old",
            "raw_path": b"/old",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
            "client": ("testclient", 50000),
        }
    )
    assert str(request.url).endswith("/old")

    normalize_request_path(request, "/new path")

    assert request.scope["path"] == "/new path"
    assert request.scope["raw_path"] == b"/new%20path"
    assert request.url.path == "/new path"
