"""Unit tests for the arm runtime.

No real proxy, network, or subprocess: ``asyncio.create_subprocess_exec`` and the httpx ready
probe are monkeypatched. The single ``@pytest.mark.live`` test opts into a real proxy spawn and
skips unless ANTHROPIC_API_KEY (or an explicit HEADROOM_LIVE) is present.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import httpx
import pytest

from agent_evals.arms import (
    ArmHandle,
    HeadroomArm,
    allocate_port,
    build_arm_env,
    build_proxy_command,
)
from agent_evals.config import ProxyLaunchConfig, Settings
from agent_evals.models import ArmName, ArmSpec, Pricing, Provider, ProxyMode, TaskSavings
from agent_evals.protocols import Arm as ArmProto
from agent_evals.protocols import ArmHandle as ArmHandleProto

# --------------------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    """Settings with a deterministic, narrow proxy config for tests."""

    proxy = ProxyLaunchConfig(
        headroom_cmd=["headroom", "proxy"],
        port_range_start=18800,
        port_range_end=18810,
        readyz_path="/readyz",
        readyz_timeout_s=1.0,
        poll_interval_s=0.01,
    )
    base: dict[str, object] = {"proxy": proxy}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _spec(
    name: ArmName,
    provider: Provider,
    proxy_mode: ProxyMode | None,
    proxy_flags: list[str] | None = None,
) -> ArmSpec:
    return ArmSpec(
        name=name,
        provider=provider,
        proxy_mode=proxy_mode,
        proxy_flags=proxy_flags or [],
        label=name.value,
    )


class _FakeProcess:
    """Stand-in for asyncio.subprocess.Process: records terminate/kill, controls returncode."""

    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self._wait_returns = 0

    def terminate(self) -> None:
        self.terminated = True
        # A well-behaved proxy exits on SIGTERM.
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._wait_returns
        return self.returncode


# --------------------------------------------------------------------------------------------
# build_proxy_command — PURE
# --------------------------------------------------------------------------------------------


def test_build_proxy_command_a0_guards() -> None:
    """A0 (proxy_mode=None) must raise — it never launches a proxy."""

    spec = _spec(ArmName.A0_DIRECT, Provider.ANTHROPIC, proxy_mode=None)
    with pytest.raises(ValueError, match="A0 direct launches no proxy"):
        build_proxy_command(spec, _settings(), port=18800)


def test_build_proxy_command_a1_passthrough() -> None:
    """A1 OFF mode -> --no-optimize, exact argv."""

    spec = _spec(ArmName.A1_PASSTHROUGH, Provider.ANTHROPIC, proxy_mode=ProxyMode.OFF)
    assert build_proxy_command(spec, _settings(), port=18801) == [
        "headroom",
        "proxy",
        "--port",
        "18801",
        "--no-optimize",
    ]


def test_build_proxy_command_b_token() -> None:
    """B TOKEN mode -> --mode token, exact argv."""

    spec = _spec(ArmName.B_HEADROOM, Provider.ANTHROPIC, proxy_mode=ProxyMode.TOKEN)
    assert build_proxy_command(spec, _settings(), port=18802) == [
        "headroom",
        "proxy",
        "--port",
        "18802",
        "--mode",
        "token",
    ]


def test_build_proxy_command_ablation_appends_flags() -> None:
    """Ablation arm: extra proxy_flags appended verbatim after the mode flag."""

    spec = _spec(
        ArmName.B_ABLATE,
        Provider.ANTHROPIC,
        proxy_mode=ProxyMode.TOKEN,
        proxy_flags=["--disable-kompress", "--no-read-lifecycle"],
    )
    assert build_proxy_command(spec, _settings(), port=18803) == [
        "headroom",
        "proxy",
        "--port",
        "18803",
        "--mode",
        "token",
        "--disable-kompress",
        "--no-read-lifecycle",
    ]


def test_build_proxy_command_honors_custom_cmd() -> None:
    """The command head is taken from settings, not hardcoded."""

    settings = _settings()
    settings.proxy.headroom_cmd = ["python", "-m", "headroom", "proxy"]
    spec = _spec(ArmName.A1_PASSTHROUGH, Provider.ANTHROPIC, proxy_mode=ProxyMode.OFF)
    assert build_proxy_command(spec, settings, port=99)[:4] == ["python", "-m", "headroom", "proxy"]


# --------------------------------------------------------------------------------------------
# build_arm_env — PURE
# --------------------------------------------------------------------------------------------


def test_build_arm_env_anthropic_no_v1_suffix() -> None:
    """Anthropic: ANTHROPIC_BASE_URL set to the root, no /v1 appended."""

    spec = _spec(ArmName.B_HEADROOM, Provider.ANTHROPIC, proxy_mode=ProxyMode.TOKEN)
    env = build_arm_env(spec, _settings(), "http://127.0.0.1:18800")
    assert env == {"ANTHROPIC_BASE_URL": "http://127.0.0.1:18800"}


def test_build_arm_env_openai_appends_v1() -> None:
    """OpenAI: both OPENAI_BASE_URL and OPENAI_API_BASE set, /v1 appended once."""

    spec = _spec(ArmName.B_HEADROOM, Provider.OPENAI, proxy_mode=ProxyMode.TOKEN)
    env = build_arm_env(spec, _settings(), "http://127.0.0.1:18800")
    assert env == {
        "OPENAI_BASE_URL": "http://127.0.0.1:18800/v1",
        "OPENAI_API_BASE": "http://127.0.0.1:18800/v1",
    }


def test_build_arm_env_openai_keeps_single_v1() -> None:
    """OpenAI: a base_url that already ends in /v1 is not double-suffixed."""

    spec = _spec(ArmName.A0_DIRECT, Provider.OPENAI, proxy_mode=None)
    env = build_arm_env(spec, _settings(), "https://api.openai.com/v1")
    assert env["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert env["OPENAI_API_BASE"] == "https://api.openai.com/v1"


# --------------------------------------------------------------------------------------------
# allocate_port
# --------------------------------------------------------------------------------------------


def test_allocate_port_in_range() -> None:
    settings = _settings()
    port = allocate_port(settings)
    assert settings.proxy.port_range_start <= port <= settings.proxy.port_range_end


def test_allocate_port_exhausted_range_raises() -> None:
    """When every port in the range is occupied, allocate_port raises a clear error."""

    # Hold a single-port range open so allocation cannot succeed.
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    busy_port = held.getsockname()[1]
    try:
        proxy = ProxyLaunchConfig(port_range_start=busy_port, port_range_end=busy_port)
        settings = Settings(proxy=proxy)
        with pytest.raises(RuntimeError, match="no free port"):
            allocate_port(settings)
    finally:
        held.close()


# --------------------------------------------------------------------------------------------
# ArmHandle
# --------------------------------------------------------------------------------------------


def test_arm_handle_capture_savings_delegates() -> None:
    sentinel = TaskSavings.from_token_counts(
        tokens_before=100, tokens_after=60, pricing=Pricing(input_usd_per_1m=3.0)
    )

    def provider(task_id: str) -> TaskSavings | None:
        return sentinel if task_id == "t1" else None

    handle = ArmHandle("http://x", {}, provider)
    assert handle.capture_savings("t1") is sentinel
    assert handle.capture_savings("other") is None


def test_arm_handle_capture_savings_none_without_provider() -> None:
    handle = ArmHandle("http://x", {})
    assert handle.capture_savings("t1") is None


def test_arm_handle_satisfies_protocol() -> None:
    handle = ArmHandle("http://x", {"K": "V"})
    assert isinstance(handle, ArmHandleProto)


# --------------------------------------------------------------------------------------------
# HeadroomArm.__aenter__ / __aexit__
# --------------------------------------------------------------------------------------------


async def test_aenter_a0_spawns_no_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A0 direct: no subprocess, provider-default base_url + env."""

    import asyncio

    async def _fail_spawn(*_a: object, **_k: object) -> object:
        raise AssertionError("create_subprocess_exec must NOT be called for A0 direct")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fail_spawn)

    settings = _settings(anthropic_base_url="https://api.anthropic.com")
    spec = _spec(ArmName.A0_DIRECT, Provider.ANTHROPIC, proxy_mode=None)
    arm = HeadroomArm(spec, settings, tmp_path)

    async with arm as handle:
        assert handle.base_url == "https://api.anthropic.com"
        assert handle.env == {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
    # No process was started -> exit is a clean no-op.
    assert arm._process is None


async def test_aenter_b_spawns_and_becomes_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B: spawn fake process, ready probe returns 200 -> localhost base_url; exit terminates."""

    import asyncio

    fake_proc = _FakeProcess(returncode=None)
    captured: dict[str, object] = {}

    async def _fake_spawn(*command: object, **kwargs: object) -> _FakeProcess:
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    class _ReadyClient:
        async def __aenter__(self) -> _ReadyClient:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def get(self, url: str, timeout: float | None = None) -> httpx.Response:
            assert url.endswith("/readyz")
            return httpx.Response(200)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _ReadyClient())

    settings = _settings()
    spec = _spec(ArmName.B_HEADROOM, Provider.ANTHROPIC, proxy_mode=ProxyMode.TOKEN)
    arm = HeadroomArm(spec, settings, tmp_path)

    async with arm as handle:
        assert handle.base_url.startswith("http://127.0.0.1:")
        port = int(handle.base_url.rsplit(":", 1)[1])
        assert settings.proxy.port_range_start <= port <= settings.proxy.port_range_end
        assert handle.env == {"ANTHROPIC_BASE_URL": handle.base_url}
        # argv was built with the right mode flag.
        assert "--mode" in captured["command"] and "token" in captured["command"]
        # A log file was opened under run_dir.
        assert any(tmp_path.glob("proxy-*.log"))

    # __aexit__ terminated the process and closed the log.
    assert fake_proc.terminated is True
    assert arm._process is None
    assert arm._log_file is None


async def test_aenter_b_readyz_timeout_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ready probe never returns 200 -> __aenter__ raises a clear RuntimeError and tears down."""

    import asyncio

    fake_proc = _FakeProcess(returncode=None)

    async def _fake_spawn(*command: object, **kwargs: object) -> _FakeProcess:
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    class _NeverReadyClient:
        async def __aenter__(self) -> _NeverReadyClient:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def get(self, url: str, timeout: float | None = None) -> httpx.Response:
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _NeverReadyClient())

    settings = _settings()
    spec = _spec(ArmName.B_HEADROOM, Provider.ANTHROPIC, proxy_mode=ProxyMode.TOKEN)
    arm = HeadroomArm(spec, settings, tmp_path)

    with pytest.raises(RuntimeError, match="did not become ready"):
        await arm.__aenter__()

    # The failed launch tore the child down (terminate called) and cleared state.
    assert fake_proc.terminated is True
    assert arm._process is None
    assert arm._log_file is None


async def test_aenter_b_process_exits_early_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the proxy exits before readyz, __aenter__ raises citing the exit code."""

    import asyncio

    fake_proc = _FakeProcess(returncode=1)  # already dead

    async def _fake_spawn(*command: object, **kwargs: object) -> _FakeProcess:
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    class _NeverGetsCalledClient:
        async def __aenter__(self) -> _NeverGetsCalledClient:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def get(self, url: str, timeout: float | None = None) -> httpx.Response:
            raise AssertionError("should not poll a dead proxy")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _NeverGetsCalledClient())

    settings = _settings()
    spec = _spec(ArmName.B_HEADROOM, Provider.ANTHROPIC, proxy_mode=ProxyMode.TOKEN)
    arm = HeadroomArm(spec, settings, tmp_path)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        await arm.__aenter__()


async def test_aexit_idempotent_for_a0(tmp_path: Path) -> None:
    """__aexit__ is safe to call when no process was ever started."""

    settings = _settings()
    spec = _spec(ArmName.A0_DIRECT, Provider.OPENAI, proxy_mode=None)
    arm = HeadroomArm(spec, settings, tmp_path)
    await arm.__aenter__()
    await arm.__aexit__(None, None, None)
    await arm.__aexit__(None, None, None)  # second call is a no-op


def test_headroom_arm_satisfies_protocol(tmp_path: Path) -> None:
    spec = _spec(ArmName.A0_DIRECT, Provider.ANTHROPIC, proxy_mode=None)
    arm = HeadroomArm(spec, _settings(), tmp_path)
    assert isinstance(arm, ArmProto)


# --------------------------------------------------------------------------------------------
# Live test (opt-in): spawn a real headroom proxy in passthrough and hit /readyz.
# --------------------------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("HEADROOM_LIVE"),
    reason="requires ANTHROPIC_API_KEY (or HEADROOM_LIVE=1) and an installed headroom",
)
async def test_live_real_proxy_readyz(tmp_path: Path) -> None:
    """Spawn `headroom proxy --no-optimize`, confirm /readyz, then tear down."""

    settings = _settings(anthropic_base_url="https://api.anthropic.com")
    settings.proxy.readyz_timeout_s = 60.0
    settings.proxy.poll_interval_s = 0.5
    spec = _spec(ArmName.A1_PASSTHROUGH, Provider.ANTHROPIC, proxy_mode=ProxyMode.OFF)
    arm = HeadroomArm(spec, settings, tmp_path)

    async with arm as handle:
        assert handle.base_url.startswith("http://127.0.0.1:")
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{handle.base_url}{settings.proxy.readyz_path}", timeout=5.0)
            assert resp.status_code == 200
    assert arm._process is None
