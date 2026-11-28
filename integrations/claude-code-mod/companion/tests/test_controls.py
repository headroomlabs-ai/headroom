from datetime import datetime, timezone

from conftest import OTHER, SID, Log, make_app
from fastapi import Request
from fastapi.testclient import TestClient

BASE = f"/headroom-mod/v1/sessions/{SID}"


def test_pause_only_bypasses_tagged_conversation_and_preserves_body(monkeypatch):
    app, _, _, _ = make_app(monkeypatch)

    @app.post("/v1/messages")
    async def upstream(request: Request):
        return {"bypass": request.headers.get("x-headroom-bypass"), "body": await request.json()}

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as c:
        assert c.post(BASE + "/compression", json={"enabled": False}).status_code == 200
        for sid, want in ((SID, "true"), (OTHER, None), (None, None)):
            headers = {"X-Headroom-Mod-Session": sid} if sid else {}
            result = c.post("/v1/messages", headers=headers, json={"messages": ["unchanged"]})
            assert result.json() == {"bypass": want, "body": {"messages": ["unchanged"]}}
        assert c.get(BASE).json()["compression_enabled"] is False
        assert c.post(BASE + "/compression", json={"enabled": True}).json()["compression_enabled"]
        assert (
            c.post("/v1/messages", headers={"X-Headroom-Mod-Session": SID}, json={}).json()[
                "bypass"
            ]
            is None
        )


def test_controls_reject_cross_origin_and_non_boolean_values(system):
    c, _, _, _ = system
    for suffix, body in (("/compression", {"enabled": False}), ("/reset", {})):
        assert (
            c.post(BASE + suffix, json=body, headers={"Origin": "https://evil.example"}).status_code
            == 403
        )
    for body in ({"enabled": "false"}, {"enabled": 0}, {}, {"enabled": False, "global": True}):
        assert c.post(BASE + "/compression", json=body).status_code == 422


def test_reset_excludes_old_requests_without_deleting_history(system):
    c, logger, _, _ = system
    assert c.get(BASE).json()["totals"]["requests"] == 1
    result = c.post(BASE + "/reset", json={})
    assert result.status_code == 200
    assert c.get(BASE).json()["totals"]["requests"] == 0
    assert c.get(BASE + "/requests/req-1").status_code == 200
    logger._logs.append(Log(request_id="new", timestamp=datetime.now(timezone.utc).isoformat()))
    assert [r["request_id"] for r in c.get(BASE).json()["requests"]] == ["new"]
    assert len(logger._logs) == 2


def test_window_filters_before_totals_and_respects_reset(system):
    c, logger, _, _ = system
    logger._logs.append(Log(request_id="now", timestamp=datetime.now(timezone.utc).isoformat()))
    assert c.get(BASE).json()["totals"]["requests"] == 2
    for window in ("15m", "1h", "24h"):
        data = c.get(BASE + f"?window={window}").json()
        assert data["totals"]["requests"] == 1
        assert data["latest"]["request_id"] == "now"
    assert c.get(BASE + "?window=forever").status_code == 422
    c.post(BASE + "/reset", json={})
    assert c.get(BASE + "?window=all").json()["totals"]["requests"] == 0


def test_pause_state_never_silently_evicted(system):
    c, _, _, _ = system
    assert c.post(BASE + "/compression", json={"enabled": False}).status_code == 200
    for n in range(300):
        sid = f"00000000-0000-4000-8000-{n:012d}"
        status = c.post(
            f"/headroom-mod/v1/sessions/{sid}/compression", json={"enabled": False}
        ).status_code
        assert status in (200, 409)
    assert c.get(BASE).json()["compression_enabled"] is False


def test_window_totals_include_more_than_the_display_limit(system):
    c, logger, _, _ = system
    now = datetime.now(timezone.utc).isoformat()
    for n in range(120):
        logger._logs.append(Log(request_id=f"recent-{n}", timestamp=now))
    logger._logs.append(Log(request_id="invalid", timestamp="unknown"))
    data = c.get(BASE + "?window=15m&limit=3").json()
    assert data["totals"]["requests"] == 120
    assert data["totals"]["saved"] == 48000
    assert len(data["requests"]) == 3
