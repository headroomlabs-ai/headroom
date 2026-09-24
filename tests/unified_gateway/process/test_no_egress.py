from __future__ import annotations

import httpx


def test_unauthenticated_request_is_rejected_without_egress(gateway_process) -> None:
    response = httpx.post(
        f"{gateway_process.base_url}/v1/responses",
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input": "secret"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "gateway_auth_required"
