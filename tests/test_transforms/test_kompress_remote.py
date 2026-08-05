import httpx

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.kompress_compressor import KompressConfig
from headroom.transforms.kompress_remote import RemoteKompressCompressor


def _long_text() -> str:
    return " ".join(f"word{i}" for i in range(20))


def _compressor(transport: httpx.BaseTransport) -> RemoteKompressCompressor:
    compressor = RemoteKompressCompressor(
        "https://kompress.example",
        token="secret",
        config=KompressConfig(enable_ccr=False),
    )
    compressor._client = httpx.Client(transport=transport)
    return compressor


def test_remote_kompress_posts_content_and_returns_result() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["json"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "compressed": "short result",
                "original_tokens": 20,
                "compressed_tokens": 2,
                "compression_ratio": 0.1,
                "model_used": "remote-model",
            },
        )

    compressor = _compressor(httpx.MockTransport(handler))
    try:
        result = compressor.compress(_long_text(), target_ratio=0.3)
    finally:
        compressor.close()

    assert seen["url"] == "https://kompress.example/compress"
    assert seen["authorization"] == "Bearer secret"
    assert '"target_ratio":0.3' in str(seen["json"]).replace(" ", "")
    assert result.compressed == "short result"
    assert result.original_tokens == 20
    assert result.compressed_tokens == 2
    assert result.compression_ratio == 0.1
    assert result.model_used == "remote-model"


def test_remote_kompress_short_input_skips_network() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"compressed": "unused"})

    compressor = _compressor(httpx.MockTransport(handler))
    try:
        result = compressor.compress("too short")
    finally:
        compressor.close()

    assert called is False
    assert result.compressed == "too short"
    assert result.compression_ratio == 1.0


def test_remote_kompress_http_error_fails_open() -> None:
    content = _long_text()
    compressor = _compressor(httpx.MockTransport(lambda request: httpx.Response(503)))
    try:
        result = compressor.compress(content)
    finally:
        compressor.close()

    assert result.compressed == content
    assert result.compression_ratio == 1.0


def test_remote_kompress_malformed_success_fails_open() -> None:
    content = _long_text()
    compressor = _compressor(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    try:
        result = compressor.compress(content)
    finally:
        compressor.close()

    assert result.compressed == content
    assert result.compression_ratio == 1.0


def test_remote_kompress_null_numeric_field_fails_open() -> None:
    # A 200 response with a valid 'compressed' but a malformed numeric field
    # (here an explicit JSON null) must still fail open, not raise. data.get
    # returns None for a present key, so float(None) would blow up if the
    # coercions were outside the fail-open guard.
    content = _long_text()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"compressed": "short result", "compression_ratio": None},
        )

    compressor = _compressor(httpx.MockTransport(handler))
    try:
        result = compressor.compress(content)
    finally:
        compressor.close()

    assert result.compressed == content
    assert result.compression_ratio == 1.0


def test_remote_kompress_non_numeric_field_fails_open() -> None:
    # A non-numeric string in a numeric field is also a malformed response.
    content = _long_text()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"compressed": "short result", "original_tokens": "lots"},
        )

    compressor = _compressor(httpx.MockTransport(handler))
    try:
        result = compressor.compress(content)
    finally:
        compressor.close()

    assert result.compressed == content
    assert result.compression_ratio == 1.0


def test_content_router_selects_remote_kompress_from_env(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "https://kompress.example")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT_TOKEN", "secret")

    router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
    compressor = router._get_kompress()
    try:
        assert isinstance(compressor, RemoteKompressCompressor)
        assert compressor.config == KompressConfig(enable_ccr=False)
        assert compressor._url == "https://kompress.example/compress"
        assert compressor._headers["authorization"] == "Bearer secret"
    finally:
        compressor.close()
