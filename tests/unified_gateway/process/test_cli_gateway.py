from __future__ import annotations

import httpx


def test_real_gateway_process_reports_local_readiness(gateway_process) -> None:
    response = httpx.get(f"{gateway_process.base_url}/readyz")
    assert response.status_code == 200
    assert response.json()["profile"] == "gateway"
    assert "provider-secret" not in response.text
