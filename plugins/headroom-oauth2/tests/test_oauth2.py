import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from headroom_oauth2 import (
    OAuth2ClientCredentials,
    OAuth2Error,
    OAuth2Middleware,
    _split,
    install,
    parse_headers,
    provider_from_env,
)


class _IdP(BaseHTTPRequestHandler):
    last_form = None
    last_auth = None
    status = 200
    tok = "TOK-1"
    expires_in = 3600
    mint_count = 0
    slow = False
    non_json = False
    omit_expires = False

    def do_POST(self):
        n = int(self.headers.get("content-length", 0) or 0)
        _IdP.last_form = self.rfile.read(n).decode()
        _IdP.last_auth = self.headers.get("authorization")
        if _IdP.status != 200:
            self.send_response(_IdP.status)
            self.end_headers()
            self.wfile.write(b'{"error":"bad","error_description":"SENSITIVE"}')
            return
        if _IdP.slow:
            time.sleep(0.05)  # widen the window so concurrent callers contend on the lock
        _IdP.mint_count += 1
        if _IdP.non_json:
            body = b"<html>not json SENSITIVE</html>"
        else:
            payload = {"access_token": _IdP.tok, "token_type": "Bearer"}
            if not _IdP.omit_expires:
                payload["expires_in"] = _IdP.expires_in
            body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def idp():
    _IdP.last_form = _IdP.last_auth = None
    _IdP.status = 200
    _IdP.tok = "TOK-1"
    _IdP.expires_in = 3600
    _IdP.mint_count = 0
    _IdP.slow = False
    _IdP.non_json = False
    _IdP.omit_expires = False
    srv = HTTPServer(("127.0.0.1", 0), _IdP)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/token"
    srv.shutdown()


# --- minimal ASGI test doubles -------------------------------------------------
class _RecordingApp:
    def __init__(self):
        self.called = False
        self.scope = None

    async def __call__(self, scope, receive, send):
        self.called = True
        self.scope = scope


async def _recv():
    return {"type": "http.request"}


async def _ignore(_msg):
    pass


def _cfg(backend):
    return type("Cfg", (), {"backend": backend})()


# --- provider: token minting ---------------------------------------------------
def test_post_style_mint(idp):
    p = OAuth2ClientCredentials(
        token_url=idp, client_id="cid", client_secret="csec", scopes=["a", "b"], audience="aud"
    )
    assert p.token() == "TOK-1"
    assert "grant_type=client_credentials" in _IdP.last_form
    assert "scope=a+b" in _IdP.last_form
    assert "client_id=cid" in _IdP.last_form
    assert "audience=aud" in _IdP.last_form
    assert _IdP.last_auth is None


def test_basic_style_mint(idp):
    p = OAuth2ClientCredentials(
        token_url=idp, client_id="cid", client_secret="csec", auth_style="basic"
    )
    p.token()
    assert _IdP.last_auth == "Basic " + base64.b64encode(b"cid:csec").decode()
    assert "client_secret" not in _IdP.last_form


def test_cache_and_refresh(idp):
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    assert p.token() == "TOK-1"
    _IdP.tok = "TOK-2"
    _IdP.last_form = None
    assert p.token() == "TOK-1"  # cached -> no re-mint
    assert _IdP.last_form is None
    p._exp = time.monotonic() - 1  # force expiry
    assert p.token() == "TOK-2"  # re-minted


def test_cached_fast_path(idp):
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    assert p.cached() is None  # nothing minted yet -> middleware will mint off-loop
    p.token()
    assert p.cached() == "TOK-1"  # now served without a token endpoint round-trip
    p._exp = time.monotonic() - 1
    assert p.cached() is None  # expired -> forces a refresh


def test_concurrent_single_flight(idp):
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    _IdP.slow = True
    out = []
    threads = [threading.Thread(target=lambda: out.append(p.token())) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == ["TOK-1"] * 12
    assert _IdP.mint_count == 1  # 12 concurrent callers -> exactly one mint


# --- provider: failure modes ---------------------------------------------------
def test_error_on_bad_status_hides_body(idp):
    _IdP.status = 401
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    with pytest.raises(OAuth2Error) as ei:
        p.token()
    assert "SENSITIVE" not in str(ei.value)  # IdP error body must not leak into the exception


def test_malformed_200_no_token(idp):
    _IdP.tok = None  # HTTP 200 but no access_token field
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    with pytest.raises(OAuth2Error):
        p.token()


def test_unreachable_token_url():
    p = OAuth2ClientCredentials(
        token_url="http://127.0.0.1:1/token", client_id="c", client_secret="s", timeout_seconds=1
    )
    with pytest.raises(OAuth2Error):
        p.token()


def test_validation():
    with pytest.raises(ValueError):
        OAuth2ClientCredentials(token_url="", client_id="c", client_secret="s")
    with pytest.raises(ValueError):
        OAuth2ClientCredentials(token_url="u", client_id="", client_secret="s")
    with pytest.raises(ValueError):
        OAuth2ClientCredentials(token_url="u", client_id="c", client_secret="s", auth_style="x")


def test_https_enforced():
    with pytest.raises(ValueError):
        OAuth2ClientCredentials(
            token_url="http://example.com/token", client_id="c", client_secret="s"
        )
    # loopback http allowed for local testing
    OAuth2ClientCredentials(token_url="http://127.0.0.1:1/token", client_id="c", client_secret="s")
    # explicit opt-out
    OAuth2ClientCredentials(
        token_url="http://example.com/token", client_id="c", client_secret="s", allow_insecure=True
    )


def test_expires_in_clamp(idp):
    _IdP.expires_in = 0  # immediate-expiry -> must clamp to a positive ttl (not stale)
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    assert p.token() == "TOK-1"
    assert p._exp > time.monotonic()

    _IdP.expires_in = -10  # negative -> must clamp (not perpetual re-mint)
    p2 = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    assert p2.token() == "TOK-1"
    assert p2._exp > time.monotonic()


# --- helpers / config ----------------------------------------------------------
def test_helpers():
    assert _split("a, b  c") == ["a", "b", "c"]
    assert parse_headers("X=1,Y=2") == {"X": "1", "Y": "2"}


def test_parse_headers_rejects_control_chars():
    assert parse_headers("Good=ok,Bad=line\r\ninject") == {"Good": "ok"}  # CRLF value dropped
    assert parse_headers("=novalue,K=v") == {"K": "v"}  # empty key dropped
    assert parse_headers("") == {}


def test_provider_from_env_wires_knobs(idp):
    env = {
        "HEADROOM_OAUTH2_TOKEN_URL": idp,
        "HEADROOM_OAUTH2_CLIENT_ID": "c",
        "HEADROOM_OAUTH2_CLIENT_SECRET": "s",
        "HEADROOM_OAUTH2_RESOURCE": "https://api.example",
        "HEADROOM_OAUTH2_TIMEOUT": "5",
        "HEADROOM_OAUTH2_SKEW": "10",
    }
    p = provider_from_env(env)
    assert p.extra_params == {"resource": "https://api.example"}
    assert p.timeout == 5.0
    assert p.skew == 10
    p.token()
    assert "resource=https" in _IdP.last_form


def test_provider_from_env_none_when_unset():
    assert provider_from_env({}) is None


# --- middleware (ASGI behavior) ------------------------------------------------
def test_middleware_injects_bearer(idp):
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    app = _RecordingApp()
    mw = OAuth2Middleware(app, p)
    scope = {"type": "http", "headers": [(b"authorization", b"Bearer CLIENT"), (b"x-keep", b"1")]}
    asyncio.run(mw(scope, _recv, _ignore))
    hdrs = dict(app.scope["headers"])
    assert hdrs[b"authorization"] == b"Bearer TOK-1"  # client creds replaced by minted token
    assert hdrs[b"x-keep"] == b"1"  # other headers preserved


def test_middleware_non_http_passthrough():
    app = _RecordingApp()
    mw = OAuth2Middleware(app, provider=object())  # provider must never be touched
    scope = {"type": "lifespan"}
    asyncio.run(mw(scope, _recv, _ignore))
    assert app.called and app.scope is scope


def test_middleware_502_on_mint_failure():
    class _Bad:
        def cached(self):
            return None

        def token(self):
            raise OAuth2Error("nope")

    app = _RecordingApp()
    mw = OAuth2Middleware(app, _Bad())
    sent = []

    async def send(msg):
        sent.append(msg)

    asyncio.run(mw({"type": "http", "headers": []}, _recv, send))
    assert not app.called  # request must not reach upstream without credentials
    assert sent[0]["status"] == 502
    assert json.loads(sent[1]["body"])["error"]["type"] == "upstream_auth_error"


# --- install() (entry point) ---------------------------------------------------
def test_install_noop_when_unset(monkeypatch):
    monkeypatch.delenv("HEADROOM_OAUTH2_TOKEN_URL", raising=False)

    class App:
        def add_middleware(self, *a, **k):
            raise AssertionError("must not install middleware when unconfigured")

    install(App(), _cfg("litellm-openai"))  # no raise, no add_middleware


def test_install_fail_closed_on_bad_config(monkeypatch):
    monkeypatch.setenv("HEADROOM_OAUTH2_TOKEN_URL", "https://idp.example.com/token")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_ID", "")  # missing -> ValueError -> RuntimeError
    with pytest.raises(RuntimeError):
        install(object(), _cfg("litellm-openai"))


def test_install_warns_for_envauth_backend(monkeypatch, caplog):
    monkeypatch.setenv("HEADROOM_OAUTH2_TOKEN_URL", "https://idp.example.com/token")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_ID", "c")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_SECRET", "s")
    monkeypatch.delenv("HEADROOM_OAUTH2_HEADERS", raising=False)
    installed = []

    class App:
        def add_middleware(self, *a, **k):
            installed.append(True)

    with caplog.at_level("WARNING"):
        install(App(), _cfg("bedrock"))
    assert installed  # still installs
    assert "NO effect" in caplog.text  # but warns the bearer is ignored by env-auth backends


def test_install_fail_closed_on_bad_timeout(monkeypatch):
    monkeypatch.setenv("HEADROOM_OAUTH2_TOKEN_URL", "https://idp.example.com/token")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_ID", "c")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_SECRET", "s")
    monkeypatch.setenv("HEADROOM_OAUTH2_TIMEOUT", "not-a-number")  # invalid -> fail closed
    with pytest.raises(RuntimeError):
        install(object(), _cfg("litellm-openai"))


def test_parse_headers_rejects_bad_keys():
    assert parse_headers("Bad Key=v,Ok=1") == {"Ok": "1"}  # space in key dropped
    assert parse_headers("X:Y=v,Ok=1") == {"Ok": "1"}  # colon in key dropped


def test_expires_in_float(idp):
    _IdP.expires_in = 3599.9  # some IdPs return a JSON float -> must not fall back to 300
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    assert p.token() == "TOK-1"
    assert p._exp - time.monotonic() > 1000  # ~3599, not the 300 fallback


def test_middleware_handles_missing_headers_key(idp):
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    app = _RecordingApp()
    mw = OAuth2Middleware(app, p)
    asyncio.run(mw({"type": "http"}, _recv, _ignore))  # scope without a "headers" key
    assert dict(app.scope["headers"])[b"authorization"] == b"Bearer TOK-1"


def test_middleware_502_sets_no_store():
    class _Bad:
        def cached(self):
            return None

        def token(self):
            raise OAuth2Error("nope")

    app = _RecordingApp()
    mw = OAuth2Middleware(app, _Bad())
    sent = []

    async def send(msg):
        sent.append(msg)

    asyncio.run(mw({"type": "http", "headers": []}, _recv, send))
    hdrs = dict(sent[0]["headers"])
    assert hdrs[b"cache-control"] == b"no-store"  # a fronting cache must not pin the 502


# --- TLS / loopback edge cases -------------------------------------------------
def test_localhost_rejected():
    # "localhost" is a name (DNS-rebinding / /etc/hosts risk) -> not a loopback exception
    with pytest.raises(ValueError):
        OAuth2ClientCredentials(
            token_url="http://localhost/token", client_id="c", client_secret="s"
        )


def test_ipv6_loopback_allowed():
    OAuth2ClientCredentials(token_url="http://[::1]:1/token", client_id="c", client_secret="s")


def test_allow_insecure_env_permits_nonloopback_http():
    p = provider_from_env(
        {
            "HEADROOM_OAUTH2_TOKEN_URL": "http://example.com/token",
            "HEADROOM_OAUTH2_CLIENT_ID": "c",
            "HEADROOM_OAUTH2_CLIENT_SECRET": "s",
            "HEADROOM_OAUTH2_ALLOW_INSECURE": "1",
        }
    )
    assert p.token_url == "http://example.com/token"


# --- token-request form edge cases ---------------------------------------------
def test_extra_params_cannot_override_canonical(idp):
    p = OAuth2ClientCredentials(
        token_url=idp,
        client_id="cid",
        client_secret="csec",
        scopes=["a", "b"],
        extra_params={"grant_type": "evil", "client_id": "evil", "scope": "evil", "resource": "r"},
    )
    p.token()
    assert "grant_type=client_credentials" in _IdP.last_form
    assert "client_id=cid" in _IdP.last_form
    assert "scope=a+b" in _IdP.last_form
    assert "resource=r" in _IdP.last_form  # a benign extra still passes through
    assert "evil" not in _IdP.last_form  # caller extras never clobber canonical fields


def test_auth_style_basic_via_env(idp):
    p = provider_from_env(
        {
            "HEADROOM_OAUTH2_TOKEN_URL": idp,
            "HEADROOM_OAUTH2_CLIENT_ID": "cid",
            "HEADROOM_OAUTH2_CLIENT_SECRET": "csec",
            "HEADROOM_OAUTH2_AUTH_STYLE": "basic",
        }
    )
    assert p.auth_style == "basic"
    p.token()
    assert _IdP.last_auth == "Basic " + base64.b64encode(b"cid:csec").decode()


def test_scopes_comma_separated_via_env(idp):
    p = provider_from_env(
        {
            "HEADROOM_OAUTH2_TOKEN_URL": idp,
            "HEADROOM_OAUTH2_CLIENT_ID": "c",
            "HEADROOM_OAUTH2_CLIENT_SECRET": "s",
            "HEADROOM_OAUTH2_SCOPES": "a, b ,c",
        }
    )
    assert p.scopes == ["a", "b", "c"]


# --- expires_in edge cases -----------------------------------------------------
def test_expires_in_missing_falls_back(idp):
    _IdP.omit_expires = True  # no expires_in field -> default ttl, not stale
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    assert p.token() == "TOK-1"
    assert 100 < p._exp - time.monotonic() <= 300


def test_expires_in_non_numeric_falls_back(idp):
    _IdP.expires_in = "not-a-number"  # garbage -> default ttl, no crash
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    assert p.token() == "TOK-1"
    assert 100 < p._exp - time.monotonic() <= 300


# --- response-shape failure modes ----------------------------------------------
def test_non_json_200_raises_without_leak(idp):
    _IdP.non_json = True  # HTTP 200 but body is not JSON
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    with pytest.raises(OAuth2Error) as ei:
        p.token()
    assert "SENSITIVE" not in str(ei.value)  # body must not leak into the exception


def test_single_flight_on_refresh(idp):
    p = OAuth2ClientCredentials(token_url=idp, client_id="c", client_secret="s")
    p.token()  # cold mint #1
    assert _IdP.mint_count == 1
    p._exp = time.monotonic() - 1  # force expiry
    _IdP.slow = True
    threads = [threading.Thread(target=p.token) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _IdP.mint_count == 2  # exactly one refresh despite 8 concurrent expired callers


def test_install_sets_static_headers(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("litellm")  # avoid importing the real (heavy) litellm
    fake.headers = {}
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setenv("HEADROOM_OAUTH2_TOKEN_URL", "https://idp.example.com/token")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_ID", "c")
    monkeypatch.setenv("HEADROOM_OAUTH2_CLIENT_SECRET", "s")
    monkeypatch.setenv("HEADROOM_OAUTH2_HEADERS", "X-App=demo,Bad Key=x")

    class App:
        def add_middleware(self, *a, **k):
            pass

    install(App(), _cfg("litellm-openai"))
    assert fake.headers == {"X-App": "demo"}  # valid header set on litellm; malformed key dropped
