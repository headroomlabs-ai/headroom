import base64
import json

import httpx
import pytest

from headroom.providers.codex import (
    CodexRoutingDecision as ExportedCodexRoutingDecision,
)
from headroom.providers.codex import (
    resolve_codex_routing as exported_resolve_codex_routing,
)
from headroom.providers.codex.model_metadata import (
    codex_model_registry_entry,
    fetch_chatgpt_codex_model_ids,
    normalize_codex_registry_headers,
    synthetic_model_get_response,
    synthetic_models_list_response,
)
from headroom.providers.codex.runtime import (
    decode_openai_bearer_payload,
    resolve_codex_routing,
    resolve_codex_routing_headers,
)


def _jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


def test_codex_runtime_resolves_chatgpt_account_from_openai_oauth_jwt() -> None:
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )

    payload = decode_openai_bearer_payload({"authorization": f"Bearer {token}"})
    headers, is_chatgpt_auth = resolve_codex_routing_headers({"authorization": f"Bearer {token}"})

    assert payload == {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acct-from-jwt",
        }
    }
    assert is_chatgpt_auth is True
    assert headers["ChatGPT-Account-ID"] == "acct-from-jwt"


def test_codex_runtime_exposes_named_routing_decision() -> None:
    decision = resolve_codex_routing({"ChatGPT-Account-ID": "acct-explicit"})

    assert decision.is_chatgpt_auth is True
    assert decision.headers == {"ChatGPT-Account-ID": "acct-explicit"}


def test_codex_package_exports_routing_decision() -> None:
    decision = exported_resolve_codex_routing({"ChatGPT-Account-ID": "acct-root"})

    assert isinstance(decision, ExportedCodexRoutingDecision)
    assert decision.is_chatgpt_auth is True
    assert decision.headers == {"ChatGPT-Account-ID": "acct-root"}


def test_codex_runtime_ignores_malformed_oauth_jwt() -> None:
    assert decode_openai_bearer_payload({"authorization": "Bearer header.***.sig"}) is None

    headers, is_chatgpt_auth = resolve_codex_routing_headers(
        {"authorization": "Bearer header.***.sig"}
    )

    assert is_chatgpt_auth is False
    assert headers == {"authorization": "Bearer header.***.sig"}


def test_codex_registry_headers_normalize_account_and_accept_headers() -> None:
    headers = normalize_codex_registry_headers(
        {
            "Host": "localhost:8787",
            "authorization": "Bearer token",
            "ChatGPT-Account-ID": "acct",
            "Accept": "text/event-stream",
        }
    )

    assert headers == {
        "authorization": "Bearer token",
        "chatgpt-account-id": "acct",
        "accept": "application/json",
    }


@pytest.mark.asyncio
async def test_codex_model_registry_fetch_returns_slugs() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, str]]] = []

        async def get(self, url, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((url, dict(kwargs.get("headers", {}))))
            return httpx.Response(
                200,
                json={"models": [{"slug": "gpt-5.5"}, {"slug": ""}, {"slug": "gpt-5.4"}]},
            )

    client = FakeClient()

    model_ids = await fetch_chatgpt_codex_model_ids(
        client,
        {"authorization": "Bearer token", "chatgpt-account-id": "acct"},
        "0.135.0",
    )

    assert model_ids == ("gpt-5.5", "gpt-5.4")
    assert client.calls == [
        (
            "https://chatgpt.com/backend-api/codex/models?client_version=0.135.0",
            {
                "authorization": "Bearer token",
                "chatgpt-account-id": "acct",
                "accept": "application/json",
            },
        )
    ]


def test_codex_synthetic_model_metadata_responses() -> None:
    list_payload = json.loads(synthetic_models_list_response().body)
    known_payload = json.loads(synthetic_model_get_response("gpt-5.5").body)
    unknown = synthetic_model_get_response("gpt-99-future")

    assert list_payload["object"] == "list"
    assert "gpt-5.5" in {entry["id"] for entry in list_payload["data"]}
    assert known_payload == {
        "id": "gpt-5.5",
        "object": "model",
        "created": 0,
        "owned_by": "openai",
    }
    assert unknown.status_code == 404


def test_codex_model_registry_entry_preserves_upstream_fields_and_defaults() -> None:
    entry = codex_model_registry_entry(
        "gpt-5.3-codex-spark",
        {"display_name": "Spark", "context_window": 12345},
    )

    assert entry["slug"] == "gpt-5.3-codex-spark"
    assert entry["display_name"] == "Spark"
    assert entry["context_window"] == 12345
    assert entry["default_reasoning_level"] == "medium"
    assert entry["supports_parallel_tool_calls"] is True
    assert entry["supported_in_api"] is True
