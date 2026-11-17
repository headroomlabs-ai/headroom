"""Integration tests for Vertex native API endpoint with real API calls through the proxy.

Run with:
    export GCP_PROJECT_ID="your-project"
    export GCP_ACCESS_TOKEN="$(gcloud auth application-default print-access-token)"
    pytest tests/test_proxy_vertex_native_integration.py -rs -v

Use the *application-default* token, not `gcloud auth print-access-token`: a bare
user token is rejected by aiplatform.googleapis.com with 401 UNAUTHENTICATED.
Tokens expire hourly, so re-export before each run.

Partner (Anthropic) models must be enabled per-project in Model Garden. Cases
that the project cannot serve SKIP with an actionable reason instead of failing;
run with `-rs` so those reasons are printed.
"""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GCP_ACCESS_TOKEN") or not os.environ.get("GCP_PROJECT_ID"),
    reason="GCP_ACCESS_TOKEN or GCP_PROJECT_ID not set",
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


@pytest.fixture
def vertex_client():
    """Create test client for Vertex AI native API with optimization enabled."""
    config = ProxyConfig(
        optimize=True,  # Enable compression
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_key():
    """Get GCP access token from environment."""
    return os.environ.get("GCP_ACCESS_TOKEN")


@pytest.fixture
def project_id():
    """Get GCP project ID from environment."""
    return os.environ.get("GCP_PROJECT_ID")


def _upstream_message(response) -> str:
    """Best-effort extraction of the Vertex error message."""
    try:
        return str(response.json().get("error", {}).get("message", response.text))
    except Exception:
        return response.text


def _require_upstream_ok(response, url, location, model, publisher) -> None:
    """Fail loudly on proxy defects; skip on project-provisioning gaps.

    A non-200 here is one of two very different things, and conflating them
    makes this suite useless:

      * the proxy mangled/misrouted the request -> a real failure, fail hard
      * the caller's GCP project cannot serve this model here -> not our bug

    Partner (Anthropic) models in particular require a per-project enable step
    in Model Garden, so a project that is otherwise healthy will 403/404 on
    Claude until someone clicks through. Skip those with an actionable reason
    rather than reporting a red proxy.
    """
    if response.status_code == 200:
        return

    detail = _upstream_message(response)
    ctx = f"{publisher}/{model} @ {location}\n  URL: {url}\n  Vertex said: {detail}"

    if response.status_code == 401:
        pytest.fail(
            f"401 UNAUTHENTICATED for {ctx}\n"
            "  GCP_ACCESS_TOKEN is missing, stale (they expire hourly), or is a user\n"
            "  token where ADC is required. Refresh with:\n"
            "    export GCP_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)"
        )

    if response.status_code == 403:
        pytest.skip(
            f"403 PERMISSION_DENIED for {ctx}\n"
            "  The project cannot serve this model. Either aiplatform.googleapis.com is\n"
            "  disabled, the principal lacks roles/aiplatform.user, or (for Anthropic)\n"
            "  the model has not been enabled in Model Garden for this project."
        )

    if response.status_code == 404:
        pytest.skip(
            f"404 NOT_FOUND for {ctx}\n"
            "  This model is not available to the project at this location.\n"
            f"  For publisher '{publisher}', enable the model in Model Garden, or confirm\n"
            "  the location is in its supported-regions list. Note Gemini 3.x has no US\n"
            "  regional endpoint, and Claude 4.7+ dropped named regions for us/eu/global."
        )

    if response.status_code == 429:
        pytest.skip(f"429 RESOURCE_EXHAUSTED (quota) for {ctx}")

    if response.status_code == 400 and "not servable in region" in detail:
        pytest.skip(
            f"400 FAILED_PRECONDITION for {ctx}\n"
            "  The parametrized location/model pair is wrong -- fix the test matrix."
        )

    pytest.fail(
        f"Expected 200, got {response.status_code} for {ctx}\n"
        "  This status is not a known provisioning condition, so treat it as a proxy\n"
        "  defect (bad path rewrite, mangled body, dropped auth header) until proven\n"
        "  otherwise by reproducing the same call directly against Vertex with curl."
    )


# Location/model pairs are explicit rather than a cross-product: Vertex serves a
# given publisher model in only a subset of locations, so a cross-product
# generates combinations that can never return 200.
#
#   global       -> evergreen "-latest" aliases (global-only) + Claude
#   europe-west2 -> Gemini 3.x has no US regional endpoint; EU/APAC regions only
#   us-east5     -> Claude 4.6 and older have real regional serving here
#
# Deliberately no Gemini 2.5: it retires 2026-10-16 and should not be the thing
# this suite proves the proxy against.
@pytest.mark.parametrize(
    "location,model,publisher",
    [
        ("global", "gemini-flash-latest", "google"),
        ("global", "claude-sonnet-4-6", "anthropic"),
        ("europe-west2", "gemini-3.5-flash", "google"),
        ("us-east5", "claude-sonnet-4-6", "anthropic"),
    ],
)
class TestVertexNativeGenerateContent:
    """Test Vertex model endpoints."""

    def test_basic_generation(self, vertex_client, api_key, project_id, location, model, publisher):
        """Basic text generation works."""

        # Determine the action based on publisher
        action = "generateContent" if publisher == "google" else "rawPredict"
        url = f"/v1/projects/{project_id}/locations/{location}/publishers/{publisher}/models/{model}:{action}"

        # Prepare the payload (Anthropic Vertex requires Claude messages format for rawPredict,
        # Google Vertex requires contents format for generateContent)

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        if publisher == "google":
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "What is 2+2? Reply with just the number."}],
                    }
                ]
            }
        else:
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [
                    {"role": "user", "content": "What is 2+2? Reply with just the number."}
                ],
                "max_tokens": 100,
            }

        response = vertex_client.post(url, headers=headers, json=payload)
        _require_upstream_ok(response, url, location, model, publisher)
        data = response.json()

        if publisher == "google":
            assert "candidates" in data
            assert len(data["candidates"]) > 0
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            assert "4" in text
        else:
            assert "content" in data
            assert len(data["content"]) > 0
            text = data["content"][0]["text"]
            assert "4" in text

    def test_thinking_levels(self, vertex_client, api_key, project_id, location, model, publisher):
        """Test thinking extensions where supported."""
        action = "generateContent" if publisher == "google" else "rawPredict"
        url = f"/v1/projects/{project_id}/locations/{location}/publishers/{publisher}/models/{model}:{action}"

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        if publisher == "google":
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": "Think deeply about the number 42."}]}
                ],
                # REST spelling is thinkingBudget (camelCase); thinkingBudgetTokens
                # is silently ignored, which makes the assertion below meaningless.
                "generationConfig": {
                    "thinkingConfig": {"thinkingBudget": 512, "includeThoughts": True}
                },
            }
        else:
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{"role": "user", "content": "Think deeply about the number 42."}],
                "max_tokens": 8192,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            }

        response = vertex_client.post(url, headers=headers, json=payload)
        _require_upstream_ok(response, url, location, model, publisher)

        # A 200 alone is weak evidence: prove the thinking-configured request
        # actually produced a scored generation rather than an empty envelope.
        data = response.json()
        if publisher == "google":
            assert data.get("candidates"), (
                f"No candidates in thinking response: {json.dumps(data)[:500]}"
            )
            assert "usageMetadata" in data, f"No usageMetadata: {json.dumps(data)[:500]}"
        else:
            assert data.get("content"), f"No content in thinking response: {json.dumps(data)[:500]}"
            assert data.get("usage"), f"No usage in thinking response: {json.dumps(data)[:500]}"

    # Rest of the standard testing for compression could be added, similar to Gemini tests.
