"""Tests for the install_everywhere orchestrator."""

from __future__ import annotations

from headroom.mcp_registry.base import (
    MCPRegistrar,
    RegisterResult,
    RegisterStatus,
    ServerSpec,
)
from headroom.mcp_registry.install import (
    DEFAULT_PROXY_URL,
    build_headroom_spec,
    build_serena_spec,
    install_everywhere,
)


class _FakeRegistrar(MCPRegistrar):
    """Minimal registrar for orchestrator tests."""

    def __init__(
        self,
        name: str,
        *,
        detected: bool = True,
        register_result: RegisterResult | None = None,
    ) -> None:
        self.name = name
        self.display_name = name.title()
        self._detected = detected
        self._register_result = register_result or RegisterResult(
            RegisterStatus.REGISTERED, "ok"
        )
        self.calls: list[ServerSpec] = []

    def detect(self) -> bool:
        return self._detected

    def get_server(self, server_name: str) -> ServerSpec | None:
        return None

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        self.calls.append(spec)
        return self._register_result

    def unregister_server(self, server_name: str) -> bool:
        return True


# ----------------------------------------------------------------------
# build_headroom_spec
# ----------------------------------------------------------------------


def test_build_spec_default_proxy_no_env(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.mcp_registry.install.resolve_headroom_command",
        lambda: ["/opt/headroom/bin/headroom"],
    )
    spec = build_headroom_spec()
    assert spec.name == "headroom"
    assert spec.command == "/opt/headroom/bin/headroom"
    assert spec.args == ("mcp", "serve")
    assert spec.env == {}


def test_build_spec_custom_proxy_sets_env() -> None:
    spec = build_headroom_spec("http://127.0.0.1:9999")
    assert spec.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"}


def test_build_spec_default_url_omits_env() -> None:
    spec = build_headroom_spec(DEFAULT_PROXY_URL)
    assert spec.env == {}


def test_build_spec_falls_back_to_python_module_when_no_binary(monkeypatch) -> None:
    monkeypatch.setattr("headroom.install.runtime.shutil.which", lambda name: None)
    monkeypatch.setattr("headroom.install.runtime.sys.executable", "/usr/bin/python")

    spec = build_headroom_spec()

    assert spec.command == "/usr/bin/python"
    assert spec.args == ("-m", "headroom.cli", "mcp", "serve")
    assert spec.env == {}


def test_build_serena_spec_uses_agent_context(monkeypatch) -> None:
    # serena not on PATH → use uvx
    monkeypatch.setattr(
        "headroom.mcp_registry.install.shutil.which", lambda x: None
    )
    spec = build_serena_spec("codex")
    assert spec.name == "serena"
    assert spec.command == "uvx"
    assert spec.args == (
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server",
        "--project-from-cwd",
        "--context",
        "codex",
        "--open-web-dashboard",
        "False",
    )
    assert spec.env == {}


def test_build_serena_spec_prefers_local_binary(monkeypatch) -> None:
    # serena on PATH → use local binary
    monkeypatch.setattr(
        "headroom.mcp_registry.install.shutil.which",
        lambda x: "serena" if x == "serena" else None,
    )
    spec = build_serena_spec("claude-code")
    assert spec.name == "serena"
    assert spec.command == "serena"
    assert spec.args == (
        "start-mcp-server",
        "--project-from-cwd",
        "--context",
        "claude-code",
        "--open-web-dashboard",
        "False",
    )
    assert spec.env == {}


def test_build_serena_spec_disables_dashboard_popup_by_default() -> None:
    # Headroom installs Serena by default; the dashboard browser tab must not
    # auto-open. The flag overrides the user's serena_config.yml at startup,
    # so this holds even when the user never created a Serena config.
    for context in ("codex", "claude-code"):
        spec = build_serena_spec(context)
        idx = spec.args.index("--open-web-dashboard")
        assert spec.args[idx + 1] == "False"


# ----------------------------------------------------------------------
# install_everywhere
# ----------------------------------------------------------------------


def test_install_everywhere_calls_each_detected_registrar() -> None:
    a = _FakeRegistrar("a")
    b = _FakeRegistrar("b")
    results = install_everywhere(registrars=[a, b])
    assert set(results) == {"a", "b"}
    assert results["a"].status == RegisterStatus.REGISTERED
    assert results["b"].status == RegisterStatus.REGISTERED
    assert len(a.calls) == 1
    assert len(b.calls) == 1


def test_install_everywhere_skips_undetected() -> None:
    detected = _FakeRegistrar("a", detected=True)
    missing = _FakeRegistrar("b", detected=False)
    results = install_everywhere(registrars=[detected, missing])
    assert results["a"].status == RegisterStatus.REGISTERED
    assert results["b"].status == RegisterStatus.NOT_DETECTED
    assert len(detected.calls) == 1
    assert len(missing.calls) == 0


def test_install_everywhere_filters_by_agents() -> None:
    a = _FakeRegistrar("a")
    b = _FakeRegistrar("b")
    c = _FakeRegistrar("c")
    results = install_everywhere(registrars=[a, b, c], agents=["a", "c"])
    assert set(results) == {"a", "c"}
    assert "b" not in results


def test_install_everywhere_passes_proxy_url_into_spec() -> None:
    captured: list[ServerSpec] = []

    class CapturingRegistrar(_FakeRegistrar):
        def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
            captured.append(spec)
            return RegisterResult(RegisterStatus.REGISTERED, "ok")

    reg = CapturingRegistrar("x")
    install_everywhere(proxy_url="http://localhost:9000", registrars=[reg])
    assert len(captured) == 1
    assert captured[0].env == {"HEADROOM_PROXY_URL": "http://localhost:9000"}


def test_install_everywhere_passes_force_flag() -> None:
    captured: list[bool] = []

    class CapturingRegistrar(_FakeRegistrar):
        def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
            captured.append(force)
            return RegisterResult(RegisterStatus.REGISTERED, "ok")

    reg = CapturingRegistrar("x")
    install_everywhere(registrars=[reg], force=True)
    assert captured == [True]


def test_install_everywhere_returns_mismatch_results() -> None:
    mismatched = _FakeRegistrar(
        "a",
        register_result=RegisterResult(RegisterStatus.MISMATCH, "env differs"),
    )
    results = install_everywhere(registrars=[mismatched])
    assert results["a"].status == RegisterStatus.MISMATCH
    assert results["a"].ok is False  # mismatch is NOT a success
