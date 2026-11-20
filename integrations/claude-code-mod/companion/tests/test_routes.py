from conftest import OTHER, SID, Log, make_app
from fastapi.testclient import TestClient

BASE = "/headroom-mod/v1"


def test_health_identifies_contract_and_is_no_store(system):
    client, _, calls, _ = system
    r = client.get(BASE + "/health")
    assert r.status_code == 200
    assert r.json()["service"] == "headroom-claude-mod"
    assert r.headers["cache-control"] == "no-store"
    assert calls == ["peer-and-host", "origin"]


def test_filter_precedes_metadata_and_body_reads(system):
    client, logger, _, _ = system
    logger._logs.append(
        Log(
            request_id="other",
            tags={"mod-session": OTHER},
            request_messages=[{"content": "OTHER_SESSION_SECRET"}],
        )
    )
    s = client.get(f"{BASE}/sessions/{SID}").json()
    assert s["totals"]["requests"] == 1
    assert [r["request_id"] for r in s["requests"]] == ["req-1"]
    assert "OTHER_SESSION_SECRET" not in str(s)
    assert client.get(f"{BASE}/sessions/{SID}/requests/other").status_code == 404


def test_metadata_works_without_capture(monkeypatch):
    app, _, _, _ = make_app(monkeypatch, capture=False)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as c:
        data = c.get(f"{BASE}/sessions/{SID}").json()
        assert data["totals"]["saved"] == 400
        detail = c.get(f"{BASE}/sessions/{SID}/requests/req-1").json()
        assert detail["available"] is False
        assert "--log-messages" in detail["reason"]


def test_detail_omits_completion_and_arbitrary_fields(system):
    c, _, _, _ = system
    r = c.get(f"{BASE}/sessions/{SID}/requests/req-1")
    assert r.status_code == 200
    assert "compressed tool output" in r.json()["text"]
    assert "MUST_NOT_BE_SERVED" not in r.text
    assert "request_messages" not in r.text


def test_schema_query_bounds_and_http_methods(system):
    c, _, _, _ = system
    path = f"{BASE}/sessions/{SID}"
    for suffix in ("?limit=0", "?limit=101", "?limit=-1"):
        assert c.get(path + suffix).status_code == 422
    assert c.get(f"{BASE}/sessions/not-a-uuid").status_code == 422
    assert c.post(path).status_code == 405
    assert c.get(path + "/requests/req-1?side=raw").status_code == 422
    assert c.get(path + "/requests/req-1?page=-1").status_code == 422


def test_unknown_session_is_empty_not_proxy_global(system):
    c, _, _, _ = system
    data = c.get(f"{BASE}/sessions/{OTHER}").json()
    assert data["requests"] == []
    assert data["totals"]["saved"] is None


def test_loopback_origin_dependencies_are_wired(monkeypatch):
    app, _, _, _ = make_app(monkeypatch)
    with TestClient(app, base_url="http://127.0.0.1", client=("203.0.113.8", 1)) as c:
        assert c.get(BASE + "/health").status_code == 404
    with TestClient(app, base_url="http://evil.example", client=("127.0.0.1", 1)) as c:
        assert c.get(BASE + "/health").status_code == 404
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as c:
        assert (
            c.get(f"{BASE}/sessions/{SID}", headers={"Origin": "https://evil.example"}).status_code
            == 403
        )


def test_retention_reports_possible_eviction(monkeypatch):
    app, _, _, _ = make_app(monkeypatch, capacity=2, entries=[Log(), Log(request_id="req-2")])
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as c:
        data = c.get(f"{BASE}/sessions/{SID}").json()
        assert data["retention"]["window_full"]
        assert data["basis"] == "retained_request_totals_not_unique_context_or_lifetime"
        assert data["totals"]["requests"] == 2


def test_epoch_changes_on_new_proxy(monkeypatch):
    epochs = []
    for _ in range(2):
        app, _, _, _ = make_app(monkeypatch)
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as c:
            epochs.append(c.get(BASE + "/health").json()["epoch"])
    assert epochs[0] != epochs[1]


def test_logger_drift_fails_503_not_zero(system):
    c, logger, _, _ = system
    logger._logs = []
    assert c.get(BASE + "/health").status_code == 503


def test_repeated_completion_record_not_double_counted(system):
    c, logger, _, _ = system
    logger._logs.append(
        Log(
            request_id="req-1",
            input_tokens_original=1000,
            input_tokens_optimized=500,
            tokens_saved=500,
        )
    )
    data = c.get(f"{BASE}/sessions/{SID}").json()
    assert data["totals"]["requests"] == 1
    assert data["totals"]["saved"] == 500
