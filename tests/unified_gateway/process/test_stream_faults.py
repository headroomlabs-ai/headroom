from __future__ import annotations

import httpx


def test_query_credentials_are_rejected_before_transport(gateway_process) -> None:
    response = httpx.post(
        f"{gateway_process.base_url}/v1/responses?api_key=client-secret",
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input": "secret"},
    )
    assert response.status_code == 401
