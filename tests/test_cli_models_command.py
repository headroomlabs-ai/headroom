"""`headroom models` must enumerate live and never fall back to a baked-in list.

The whole point of the command is to stop agents naming models from training
memory (`claude-3.5-sonnet`, `gemini-2.5-pro`, ... all seen live in proxy.log and
all retired). A hardcoded fallback would reintroduce exactly that failure, so the
"provider unreachable" path is asserted to produce a *reason*, not a list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from headroom.cli import models as models_cmd
from headroom.cli.main import main

FIXTURE = Path(__file__).parent / "fixtures" / "copilot_models" / "models_list.json"


@pytest.fixture(autouse=True)
def _no_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def _stub_copilot(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    class _Res:
        api_url = "https://api.githubcopilot.com"
        token = "tok"

    monkeypatch.setattr(
        "headroom.copilot_auth.resolve_subscription_bearer_token_details", lambda: _Res()
    )
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json=payload))


def test_lists_live_copilot_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_copilot(monkeypatch)
    result = CliRunner().invoke(main, ["models", "--provider", "copilot"])
    assert result.exit_code == 0, result.output
    assert "claude-opus-4.8" in result.output
    assert "mai-code-1-flash-picker" in result.output
    # Embeddings are not chat models and must not be offered.
    assert "text-embedding" not in result.output


def test_never_lists_retired_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_copilot(monkeypatch)
    result = CliRunner().invoke(main, ["models", "--provider", "copilot"])
    for retired in ("gemini-2.5-pro", "gpt-5.2", "o1-experimental", "claude-3.5-sonnet"):
        assert retired not in result.output


def test_tier_filter_surfaces_high_capability_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_copilot(monkeypatch)
    result = CliRunner().invoke(main, ["models", "--provider", "copilot", "--tier", "powerful"])
    assert result.exit_code == 0, result.output
    assert "claude-opus-5" in result.output
    assert "gpt-5.5" in result.output
    # lightweight tier must be excluded
    assert "claude-haiku-4.5" not in result.output


def test_json_output_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_copilot(monkeypatch)
    result = CliRunner().invoke(main, ["models", "--provider", "copilot", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    ids = {m["id"] for m in payload["models"]}
    assert "claude-opus-4.8" in ids
    entry = next(m for m in payload["models"] if m["id"] == "claude-opus-4.8")
    assert entry["provider"] == "copilot"
    assert entry["vendor"] == "Anthropic"


def test_unreachable_provider_reports_a_reason_not_a_fallback_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale built-in list is the bug; "unavailable" is the correct answer."""

    class _Res:
        api_url = "https://api.githubcopilot.com"
        token = "tok"

    monkeypatch.setattr(
        "headroom.copilot_auth.resolve_subscription_bearer_token_details", lambda: _Res()
    )

    def _boom(*a: Any, **k: Any) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    result = CliRunner().invoke(main, ["models", "--provider", "copilot"])
    assert result.exit_code == 0, result.output
    assert "not enumerated" in result.output
    assert "claude-opus-4.8" not in result.output


def test_anthropic_without_credential_explains_the_remedy() -> None:
    rows, reason = models_cmd._anthropic_rows()
    assert rows == []
    assert reason is not None
    assert "ANTHROPIC_API_KEY" in reason


def test_anthropic_enumerates_from_the_live_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: httpx.Response(
            200,
            json={
                "data": [
                    {"id": "claude-opus-4-5-20251101", "display_name": "Claude Opus 4.5"},
                    {"id": "claude-sonnet-4-5-20250929", "display_name": "Claude Sonnet 4.5"},
                ]
            },
        ),
    )
    rows, reason = models_cmd._anthropic_rows()
    assert reason is None
    assert {r.id for r in rows} == {"claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929"}
    assert all(r.provider == "anthropic" for r in rows)


def test_anthropic_follows_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/v1/models` is paginated; a second page must not be silently dropped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    seen_params: list[dict] = []
    pages = {
        None: {
            "data": [{"id": "claude-a", "display_name": "A"}],
            "has_more": True,
            "last_id": "claude-a",
        },
        "claude-a": {
            "data": [{"id": "claude-b", "display_name": "B"}],
            "has_more": False,
            "last_id": "claude-b",
        },
    }

    def fake_get(*_a, params=None, **_k):  # type: ignore[no-untyped-def]
        seen_params.append(dict(params or {}))
        return httpx.Response(200, json=pages[(params or {}).get("after_id")])

    monkeypatch.setattr(httpx, "get", fake_get)
    rows, reason = models_cmd._anthropic_rows()
    assert reason is None
    assert [r.id for r in rows] == ["claude-a", "claude-b"]
    assert [p.get("after_id") for p in seen_params] == [None, "claude-a"]


def test_anthropic_pagination_stops_on_a_repeated_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that keeps answering `has_more` with the same cursor must not loop,
    and the rows read so far must not be presented as the whole list."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    calls: list[int] = []

    def fake_get(*_a, **_k):  # type: ignore[no-untyped-def]
        calls.append(1)
        return httpx.Response(
            200, json={"data": [{"id": "claude-a"}], "has_more": True, "last_id": "claude-a"}
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    rows, reason = models_cmd._anthropic_rows()
    assert len(calls) == 2, "pagination did not stop on a repeated cursor"
    assert [r.id for r in rows] == ["claude-a", "claude-a"]
    assert reason is not None and "no new cursor" in reason and "may be incomplete" in reason


@pytest.mark.parametrize("last_id", [None, "", 7])
def test_anthropic_says_so_when_has_more_comes_without_a_cursor(
    monkeypatch: pytest.MonkeyPatch, last_id: object
) -> None:
    """`has_more` with no usable `last_id` stops after one request, with a note."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    calls: list[int] = []

    def fake_get(*_a, **_k):  # type: ignore[no-untyped-def]
        calls.append(1)
        body: dict[str, object] = {"data": [{"id": "claude-a"}], "has_more": True}
        if last_id is not None:
            body["last_id"] = last_id
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx, "get", fake_get)
    rows, reason = models_cmd._anthropic_rows()
    assert len(calls) == 1
    assert [r.id for r in rows] == ["claude-a"]
    assert reason is not None and "may be incomplete" in reason


def test_anthropic_last_page_has_no_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """A last page (`has_more` false) is complete even without a `last_id`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *_a, **_k: httpx.Response(
            200, json={"data": [{"id": "claude-a"}], "has_more": False}
        ),
    )
    rows, reason = models_cmd._anthropic_rows()
    assert [r.id for r in rows] == ["claude-a"]
    assert reason is None


def test_anthropic_says_so_when_it_stops_at_the_page_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncation is never silent: hitting the page cap returns a note."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(models_cmd, "_ANTHROPIC_MAX_PAGES", 2)
    cursor = iter(range(10))

    def fake_get(*_a, **_k):  # type: ignore[no-untyped-def]
        n = next(cursor)
        return httpx.Response(
            200, json={"data": [{"id": f"claude-{n}"}], "has_more": True, "last_id": f"claude-{n}"}
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    rows, reason = models_cmd._anthropic_rows()
    assert [r.id for r in rows] == ["claude-0", "claude-1"]
    assert reason is not None and "may be incomplete" in reason
