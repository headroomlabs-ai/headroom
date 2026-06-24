"""Tests for Qdrant environment-variable resolution.

Covers:
- Env var readers (host/port/url/api_key/https/prefer_grpc/grpc_port)
- Type coercion and validation (port range, bool parsing)
- ``build_qdrant_client_kwargs`` precedence (URL > host/port)
- Dataclass defaults reading env at construction time
- Explicit kwargs winning over env vars
- Public ``Memory`` class picking up env vars in ``qdrant-neo4j`` mode
"""

from __future__ import annotations

import pytest

from headroom.memory import qdrant_env
from headroom.memory.qdrant_env import (
    DEFAULT_QDRANT_GRPC_PORT,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
    build_qdrant_client_kwargs,
    qdrant_env_api_key,
    qdrant_env_grpc_port,
    qdrant_env_host,
    qdrant_env_https,
    qdrant_env_port,
    qdrant_env_prefer_grpc,
    qdrant_env_url,
)

# All HEADROOM_QDRANT_* vars are cleared before every test so that the host
# environment (e.g. a developer's local ``direnv``) cannot leak into unit tests.
_QDRANT_ENV_VARS = (
    "HEADROOM_QDRANT_URL",
    "HEADROOM_QDRANT_HOST",
    "HEADROOM_QDRANT_PORT",
    "HEADROOM_QDRANT_API_KEY",
    "HEADROOM_QDRANT_HTTPS",
    "HEADROOM_QDRANT_PREFER_GRPC",
    "HEADROOM_QDRANT_GRPC_PORT",
)


@pytest.fixture(autouse=True)
def _clear_qdrant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _QDRANT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# Env var readers
# =============================================================================


class TestEnvReaders:
    def test_host_defaults_when_unset(self) -> None:
        assert qdrant_env_host() == DEFAULT_QDRANT_HOST

    def test_host_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_HOST", "qdrant.internal")
        assert qdrant_env_host() == "qdrant.internal"

    def test_host_trims_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_HOST", "  qdrant.example  ")
        assert qdrant_env_host() == "qdrant.example"

    def test_host_empty_string_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_HOST", "")
        assert qdrant_env_host() == DEFAULT_QDRANT_HOST

    def test_port_defaults_when_unset(self) -> None:
        assert qdrant_env_port() == DEFAULT_QDRANT_PORT

    def test_port_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_PORT", "7333")
        assert qdrant_env_port() == 7333

    def test_port_rejects_non_numeric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_PORT", "not-a-port")
        with pytest.raises(ValueError, match="HEADROOM_QDRANT_PORT"):
            qdrant_env_port()

    def test_port_rejects_out_of_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_PORT", "70000")
        with pytest.raises(ValueError, match="valid port range"):
            qdrant_env_port()

    def test_port_rejects_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_PORT", "0")
        with pytest.raises(ValueError, match="valid port range"):
            qdrant_env_port()

    def test_url_none_when_unset(self) -> None:
        assert qdrant_env_url() is None

    def test_url_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_URL", "https://qdrant.cloud:6333")
        assert qdrant_env_url() == "https://qdrant.cloud:6333"

    def test_api_key_none_when_unset(self) -> None:
        assert qdrant_env_api_key() is None

    def test_api_key_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_API_KEY", "secret-token")
        assert qdrant_env_api_key() == "secret-token"

    def test_https_none_when_unset(self) -> None:
        assert qdrant_env_https() is None

    @pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "y", "on"])
    def test_https_truthy_values(self, monkeypatch: pytest.MonkeyPatch, truthy: str) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_HTTPS", truthy)
        assert qdrant_env_https() is True

    @pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "n", "off"])
    def test_https_falsy_values(self, monkeypatch: pytest.MonkeyPatch, falsy: str) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_HTTPS", falsy)
        assert qdrant_env_https() is False

    def test_https_rejects_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_HTTPS", "maybe")
        with pytest.raises(ValueError, match="Invalid boolean value"):
            qdrant_env_https()

    def test_prefer_grpc_defaults_false(self) -> None:
        assert qdrant_env_prefer_grpc() is False

    def test_prefer_grpc_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_PREFER_GRPC", "true")
        assert qdrant_env_prefer_grpc() is True

    def test_grpc_port_defaults(self) -> None:
        assert qdrant_env_grpc_port() == DEFAULT_QDRANT_GRPC_PORT

    def test_grpc_port_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_QDRANT_GRPC_PORT", "9999")
        assert qdrant_env_grpc_port() == 9999


# =============================================================================
# build_qdrant_client_kwargs
# =============================================================================


class TestBuildClientKwargs:
    def test_defaults_to_localhost(self) -> None:
        kwargs = build_qdrant_client_kwargs()
        assert kwargs == {"host": "localhost", "port": 6333}

    def test_explicit_host_and_port(self) -> None:
        kwargs = build_qdrant_client_kwargs(host="qdrant.internal", port=7333)
        assert kwargs == {"host": "qdrant.internal", "port": 7333}

    def test_url_takes_precedence_over_host_port(self) -> None:
        kwargs = build_qdrant_client_kwargs(
            url="https://xyz.cloud.qdrant.io:6333",
            host="ignored",
            port=9999,
        )
        assert kwargs == {"url": "https://xyz.cloud.qdrant.io:6333"}

    def test_empty_url_falls_back_to_host_port(self) -> None:
        kwargs = build_qdrant_client_kwargs(url="", host="h", port=1234)
        assert kwargs == {"host": "h", "port": 1234}

    def test_api_key_included_when_set(self) -> None:
        kwargs = build_qdrant_client_kwargs(api_key="secret")
        assert kwargs["api_key"] == "secret"

    def test_api_key_omitted_when_empty(self) -> None:
        kwargs = build_qdrant_client_kwargs(api_key="")
        assert "api_key" not in kwargs

    def test_https_passed_through_when_set(self) -> None:
        assert build_qdrant_client_kwargs(https=True)["https"] is True
        assert build_qdrant_client_kwargs(https=False)["https"] is False

    def test_https_omitted_when_none(self) -> None:
        kwargs = build_qdrant_client_kwargs(https=None)
        assert "https" not in kwargs

    def test_grpc_enables_only_when_preferred(self) -> None:
        kwargs = build_qdrant_client_kwargs(prefer_grpc=True, grpc_port=6334)
        assert kwargs["prefer_grpc"] is True
        assert kwargs["grpc_port"] == 6334

    def test_grpc_omitted_when_not_preferred(self) -> None:
        kwargs = build_qdrant_client_kwargs(prefer_grpc=False, grpc_port=6334)
        assert "prefer_grpc" not in kwargs
        assert "grpc_port" not in kwargs


# =============================================================================
# Dataclass defaults
# =============================================================================


def test_direct_mem0_config_defaults_read_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mem0Config (direct_mem0) should resolve env vars via default_factory."""
    from headroom.memory.backends.direct_mem0 import Mem0Config

    monkeypatch.setenv("HEADROOM_QDRANT_HOST", "custom.example")
    monkeypatch.setenv("HEADROOM_QDRANT_PORT", "7777")
    monkeypatch.setenv("HEADROOM_QDRANT_URL", "https://cloud.example:6333")
    monkeypatch.setenv("HEADROOM_QDRANT_API_KEY", "my-api-key")

    cfg = Mem0Config()
    assert cfg.qdrant_host == "custom.example"
    assert cfg.qdrant_port == 7777
    assert cfg.qdrant_url == "https://cloud.example:6333"
    assert cfg.qdrant_api_key == "my-api-key"


def test_direct_mem0_config_explicit_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.memory.backends.direct_mem0 import Mem0Config

    monkeypatch.setenv("HEADROOM_QDRANT_HOST", "env-host")
    monkeypatch.setenv("HEADROOM_QDRANT_PORT", "7777")

    cfg = Mem0Config(qdrant_host="explicit-host", qdrant_port=5555)
    assert cfg.qdrant_host == "explicit-host"
    assert cfg.qdrant_port == 5555


def test_mem0_config_defaults_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mem0Config (backends.mem0) should resolve env vars via default_factory."""
    from headroom.memory.backends.mem0 import Mem0Config

    monkeypatch.setenv("HEADROOM_QDRANT_URL", "https://tenant.qdrant.io")
    monkeypatch.setenv("HEADROOM_QDRANT_API_KEY", "k")

    cfg = Mem0Config()
    assert cfg.qdrant_url == "https://tenant.qdrant.io"
    assert cfg.qdrant_api_key == "k"


def test_proxy_memory_config_defaults_read_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.proxy.memory_handler import MemoryConfig

    monkeypatch.setenv("HEADROOM_QDRANT_HOST", "proxy.qdrant")
    monkeypatch.setenv("HEADROOM_QDRANT_PORT", "8888")

    cfg = MemoryConfig()
    assert cfg.qdrant_host == "proxy.qdrant"
    assert cfg.qdrant_port == 8888


def test_proxy_config_defaults_read_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.proxy.models import ProxyConfig

    monkeypatch.setenv("HEADROOM_QDRANT_URL", "https://shared.qdrant")
    cfg = ProxyConfig()
    assert cfg.memory_qdrant_url == "https://shared.qdrant"


# =============================================================================
# Memory.easy public API
# =============================================================================


def test_memory_easy_picks_up_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory() should pick up env vars for qdrant-neo4j backend kwargs."""
    from headroom.memory.easy import Memory

    monkeypatch.setenv("HEADROOM_QDRANT_URL", "https://hosted.qdrant.io:6333")
    monkeypatch.setenv("HEADROOM_QDRANT_API_KEY", "xoxp-1234")

    # backend="local" never touches qdrant, so we can instantiate without Docker.
    # We only verify the stored Qdrant settings, not the backend initialization.
    mem = Memory(backend="local")
    assert mem._qdrant_url == "https://hosted.qdrant.io:6333"
    assert mem._qdrant_api_key == "xoxp-1234"


def test_memory_easy_explicit_args_override_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.memory.easy import Memory

    monkeypatch.setenv("HEADROOM_QDRANT_HOST", "env-host")
    monkeypatch.setenv("HEADROOM_QDRANT_PORT", "1234")

    mem = Memory(backend="local", qdrant_host="explicit", qdrant_port=9999)
    assert mem._qdrant_host == "explicit"
    assert mem._qdrant_port == 9999


def test_memory_easy_defaults_when_no_env() -> None:
    from headroom.memory.easy import Memory

    mem = Memory(backend="local")
    assert mem._qdrant_host == "localhost"
    assert mem._qdrant_port == 6333
    assert mem._qdrant_url is None
    assert mem._qdrant_api_key is None


# =============================================================================
# Module-level public API
# =============================================================================


def test_module_exports_readers() -> None:
    """Sanity check: helper functions are importable from the module."""
    assert callable(qdrant_env.qdrant_env_host)
    assert callable(qdrant_env.qdrant_env_port)
    assert callable(qdrant_env.qdrant_env_url)
    assert callable(qdrant_env.qdrant_env_api_key)
    assert callable(qdrant_env.build_qdrant_client_kwargs)
