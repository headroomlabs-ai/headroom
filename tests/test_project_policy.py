"""Tests for pure project attribution policy helpers."""

from __future__ import annotations

from headroom.proxy.project_policy import (
    classify_project,
    split_project_path,
    with_project_prefix,
)


def test_classify_project_reads_project_header() -> None:
    assert classify_project({"x-headroom-project": " frontend "}) == "frontend"
    assert classify_project({"X-Headroom-Project": "api"}) == "api"
    assert classify_project({"user-agent": "codex"}) is None
    assert classify_project(object()) is None


def test_split_project_path_extracts_sanitized_project_and_path() -> None:
    assert split_project_path("/p/frontend/v1/messages") == ("frontend", "/v1/messages")
    assert split_project_path("/p/my%20repo/v1") == ("my repo", "/v1")
    assert split_project_path("/p/frontend") == ("frontend", "/")
    assert split_project_path("/v1/messages") == (None, "/v1/messages")
    assert split_project_path("/p/%20%20/v1") == (None, "/p/%20%20/v1")


def test_with_project_prefix_round_trips_with_split_project_path() -> None:
    url = with_project_prefix("http://127.0.0.1:8787/v1", "my repo")

    assert url == "http://127.0.0.1:8787/p/my%20repo/v1"
    assert split_project_path("/p/my%20repo/v1") == ("my repo", "/v1")
    assert with_project_prefix("http://127.0.0.1:8787/v1", " ") == ("http://127.0.0.1:8787/v1")
