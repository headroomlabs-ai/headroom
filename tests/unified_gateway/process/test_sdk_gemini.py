from __future__ import annotations

import httpx


def test_gemini_native_shape_is_rejected_locally_for_unknown_model(gateway_process) -> None:
    response = httpx.post(
        f"{gateway_process.base_url}/v1beta/models/not-granted:generateContent",
        headers={"x-goog-api-key": "client-secret"},
        json={"contents": [{"role": "user", "parts": [{"text": "must-not-egress"}]}]},
    )
    assert response.status_code == 404
