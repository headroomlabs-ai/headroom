"""Run this gate in the SAME environment as the real Headroom installation.

No fixture substitutes Headroom's guards, RequestLog or RequestLogger here.
This test does not contact a model provider or assert full compression-pipeline E2E.
"""

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("headroom") is None,
    reason="real Headroom is not installed in this build environment",
)


def test_with_real_request_logger_and_security_guards():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from headroom_claude_mod.extension import install

    from headroom.proxy.models import RequestLog
    from headroom.proxy.request_logger import RequestLogger

    sid = "11111111-1111-4111-8111-111111111111"
    logger = RequestLogger(log_full_messages=True)
    logger.log(
        RequestLog(
            request_id="contract-1",
            timestamp="2026-10-05T00:00:00Z",
            provider="anthropic",
            model="fixture",
            input_tokens_original=100,
            input_tokens_optimized=60,
            output_tokens=10,
            tokens_saved=40,
            savings_percent=40,
            optimization_latency_ms=1,
            total_latency_ms=2,
            tags={"mod-session": sid},
            cache_hit=False,
            transforms_applied=[],
            request_messages=[{"role": "user", "content": "original"}],
            compressed_messages=[{"role": "user", "content": "compressed"}],
        )
    )
    app = FastAPI()
    app.state.proxy = SimpleNamespace(logger=logger)
    install(app, SimpleNamespace(log_full_messages=True))
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as c:
        assert c.get(f"/headroom-mod/v1/sessions/{sid}").json()["totals"]["saved"] == 40
        assert c.get(f"/headroom-mod/v1/sessions/{sid}/requests/contract-1").json()["available"]
        for origin in (
            "http://127.0.0.1:9000",
            "http://localhost",
            "https://127.0.0.1",
            "http://127.0.0.1:0",
            "http://@127.0.0.1",
            "null",
            "",
        ):
            assert (
                c.get(
                    f"/headroom-mod/v1/sessions/{sid}/requests/contract-1",
                    headers={"Origin": origin},
                ).status_code
                == 403
            )
        assert (
            c.get(
                f"/headroom-mod/v1/sessions/{sid}/requests/contract-1",
                headers={"Origin": "http://127.0.0.1"},
            ).status_code
            == 200
        )
        assert c.get(
            f"/headroom-mod/v1/sessions/{sid}", headers={"Origin": "https://evil.example"}
        ).status_code in {403, 404}
    with TestClient(app, base_url="http://127.0.0.1", client=("203.0.113.1", 1)) as c:
        assert c.get(f"/headroom-mod/v1/sessions/{sid}").status_code == 404
