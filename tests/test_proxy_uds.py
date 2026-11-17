"""Tests for serving the proxy on a Unix domain socket (`headroom proxy --uds`).

The socket transport is a plain alternative to a TCP port — see
`headroom/proxy/uds.py` for the rationale and for why it does not restore
Claude Code's Remote Control (GH #1779).
"""

from __future__ import annotations

import socket
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.proxy import proxy as proxy_cmd
from headroom.proxy.uds import (
    UDS_SUPPORTED,
    UdsError,
    _missing_ancestors,
    _require_safe_existing_parent,
    max_uds_path_length,
    prepare_uds_path,
    remove_uds_path,
    require_uds_support,
    socket_usage_lines,
)

requires_uds = pytest.mark.skipif(
    not UDS_SUPPORTED, reason="platform has no socket.AF_UNIX (Windows)"
)

try:  # `headroom.proxy.server` pulls in the compiled Rust core.
    import headroom._core  # noqa: F401

    _CORE_BUILT = True
except ImportError:  # pragma: no cover - depends on the local build
    _CORE_BUILT = False

requires_core = pytest.mark.skipif(
    not _CORE_BUILT, reason="headroom._core is not built in this environment"
)


# --------------------------------------------------------------------------
# Platform capability — runs everywhere, since the platform is a parameter.
# --------------------------------------------------------------------------


def test_require_uds_support_rejects_windows() -> None:
    """Windows has neither socket.AF_UNIX nor an asyncio UDS transport."""
    with pytest.raises(UdsError, match="unavailable on this platform"):
        require_uds_support(platform="win32")


def test_require_uds_support_accepts_posix() -> None:
    if not UDS_SUPPORTED:
        pytest.skip("AF_UNIX missing; the platform argument cannot override that")
    require_uds_support(platform="linux")


def test_sun_path_limit_is_platform_specific() -> None:
    """Linux allows 108 bytes, the BSDs and macOS 104. Guessing high truncates."""
    assert max_uds_path_length("linux") == 108
    assert max_uds_path_length("darwin") == 104


def test_cli_rejects_uds_on_windows() -> None:
    """The CLI fails fast with a readable error, not a bind-time OSError."""
    with patch("headroom.proxy.uds.UDS_SUPPORTED", False):
        result = CliRunner().invoke(proxy_cmd, ["--uds", "/tmp/headroom-test.sock"])

    assert result.exit_code != 0
    assert "Unix domain sockets" in result.output
    assert "--port instead" in result.output


# --------------------------------------------------------------------------
# Parent-directory policy — pure logic, so it runs on every platform.
# --------------------------------------------------------------------------


def test_missing_ancestors_lists_only_absent_levels(tmp_path: Path) -> None:
    """Only these get chmod 0700; anything already on disk is left alone."""
    existing = tmp_path / "existing"
    existing.mkdir()

    missing = _missing_ancestors(existing / "a" / "b")

    assert missing == [existing / "a", existing / "a" / "b"]


def test_missing_ancestors_is_empty_for_an_existing_dir(tmp_path: Path) -> None:
    assert _missing_ancestors(tmp_path) == []


class _FakeStat:
    def __init__(self, mode: int) -> None:
        self.st_mode = mode


@pytest.mark.parametrize(
    ("mode", "accepted"),
    [
        (0o700, True),  # owner only
        (0o750, True),  # group may read/traverse, not write
        (0o755, True),  # the common shared-parent case
        (0o770, False),  # any group member could swap the socket
        (0o777, False),  # any local user could
        (0o1777, True),  # /tmp: sticky, so others cannot unlink ours
        (0o1770, True),  # sticky group-writable
    ],
)
def test_existing_parent_accepted_only_when_others_cannot_swap_the_socket(
    tmp_path: Path, mode: int, accepted: bool
) -> None:
    """Windows chmod is a no-op, so the mode is injected rather than applied."""
    with patch.object(Path, "stat", return_value=_FakeStat(stat.S_IFDIR | mode)):
        if accepted:
            _require_safe_existing_parent(tmp_path)
        else:
            with pytest.raises(UdsError, match="writable by other users"):
                _require_safe_existing_parent(tmp_path)


def test_unreadable_existing_parent_defers_to_bind(tmp_path: Path) -> None:
    """A stat we cannot perform is not evidence of a problem; let bind() rule."""
    with patch.object(Path, "stat", side_effect=PermissionError):
        _require_safe_existing_parent(tmp_path)


# --------------------------------------------------------------------------
# Startup banner — a socket bind must not advertise a broken recipe.
# --------------------------------------------------------------------------


def test_socket_usage_lines_omit_the_unsupported_claude_code_recipe() -> None:
    """Regression: the banner once printed a configuration that cannot work.

    `ANTHROPIC_UNIX_SOCKET=... claude` passes Claude Code's api.anthropic.com
    host check but reclassifies the session as API-key auth, and the session
    then fails to authenticate. Printing it at startup turned a known-negative
    field result into first-party runtime guidance.
    """
    rendered = "\n".join(socket_usage_lines("/run/headroom/proxy.sock"))

    assert "ANTHROPIC_UNIX_SOCKET" not in rendered
    assert "ANTHROPIC_BASE_URL" not in rendered
    assert "claude" not in rendered.lower()


def test_socket_usage_lines_state_the_transport_requirement() -> None:
    """What replaces the recipe has to be useful, not merely absent."""
    path = "/run/headroom/proxy.sock"

    rendered = "\n".join(socket_usage_lines(path))

    assert path in rendered
    assert "HTTP over a Unix socket" in rendered
    assert "curl --unix-socket" in rendered
    assert "serving-on-a-unix-socket" in rendered


def test_socket_usage_lines_name_no_agent() -> None:
    """Transport-neutral: the banner singles out no client."""
    rendered = "\n".join(socket_usage_lines("/run/headroom/proxy.sock")).lower()

    for agent in ("claude", "codex", "opencode", "cursor", "aider", "copilot"):
        assert agent not in rendered, f"banner should not name {agent}"


# --------------------------------------------------------------------------
# Path preparation — needs a real AF_UNIX platform.
# --------------------------------------------------------------------------


@requires_uds
def test_prepare_creates_parent_owner_only(tmp_path: Path) -> None:
    """The directory mode is the access-control boundary for the socket."""
    target = tmp_path / "run" / "headroom.sock"

    resolved = prepare_uds_path(target)

    assert resolved == target
    assert target.parent.is_dir()
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


@requires_uds
def test_prepare_preserves_an_existing_parents_mode(tmp_path: Path) -> None:
    """A caller-owned directory must not be silently tightened to 0700.

    Regression test: `--uds /run/shared/hr.sock` where `/run/shared` is a
    directory someone else set up at 0755 would have locked out every other
    user of that directory.
    """
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o755)
    bystander = parent / "someone-elses.txt"
    bystander.write_text("theirs", encoding="utf-8")

    prepare_uds_path(parent / "headroom.sock")

    assert stat.S_IMODE(parent.stat().st_mode) == 0o755
    assert bystander.read_text(encoding="utf-8") == "theirs"


@requires_uds
def test_prepare_only_chmods_directories_it_creates(tmp_path: Path) -> None:
    """The 0700 applies to the new levels, not to the existing root above them."""
    root = tmp_path / "existing"
    root.mkdir()
    root.chmod(0o755)

    prepare_uds_path(root / "a" / "b" / "headroom.sock")

    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "a").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "a" / "b").stat().st_mode) == 0o700


@requires_uds
def test_prepare_refuses_a_world_writable_existing_parent(tmp_path: Path) -> None:
    """Without the sticky bit, any local user could swap the socket out."""
    parent = tmp_path / "open"
    parent.mkdir()
    parent.chmod(0o777)

    with pytest.raises(UdsError, match="writable by other users"):
        prepare_uds_path(parent / "headroom.sock")

    assert stat.S_IMODE(parent.stat().st_mode) == 0o777, "the refusal must not mutate"


@requires_uds
def test_prepare_accepts_a_sticky_world_writable_parent(tmp_path: Path) -> None:
    """`/tmp` is 1777: others can add entries but cannot unlink ours."""
    parent = tmp_path / "sticky"
    parent.mkdir()
    parent.chmod(0o1777)

    resolved = prepare_uds_path(parent / "headroom.sock")

    assert resolved.parent == parent
    assert stat.S_IMODE(parent.stat().st_mode) == 0o1777


@requires_uds
def test_prepare_clears_a_stale_socket(tmp_path: Path) -> None:
    """A crashed proxy leaves an inode behind; a restart must not trip on it."""
    target = tmp_path / "stale.sock"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(target))
    dead.close()  # closing without unlinking is exactly the crash case
    assert target.exists()

    prepare_uds_path(target)

    assert not target.exists()


@requires_uds
def test_prepare_refuses_a_live_socket(tmp_path: Path) -> None:
    """Two proxies on one socket would silently steal each other's traffic."""
    target = tmp_path / "live.sock"
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(target))
    live.listen(1)
    try:
        with pytest.raises(UdsError, match="already listening"):
            prepare_uds_path(target)
        assert target.exists(), "the live socket must survive the refusal"
    finally:
        live.close()
        target.unlink(missing_ok=True)


@requires_uds
def test_prepare_never_deletes_a_regular_file(tmp_path: Path) -> None:
    """A typo'd --uds pointing at real data must not destroy it."""
    target = tmp_path / "notes.txt"
    target.write_text("important", encoding="utf-8")

    with pytest.raises(UdsError, match="is not a socket"):
        prepare_uds_path(target)

    assert target.read_text(encoding="utf-8") == "important"


@requires_uds
def test_prepare_rejects_an_oversized_path(tmp_path: Path) -> None:
    """Past sun_path, bind() fails with an ENAMETOOLONG that names nothing."""
    target = tmp_path / ("d" * 120) / "headroom.sock"

    with pytest.raises(UdsError, match="sun_path limit"):
        prepare_uds_path(target)


@requires_uds
def test_remove_uds_path_is_socket_only(tmp_path: Path) -> None:
    """Cleanup runs in a finally block, so it must be narrow and never raise."""
    sock_path = tmp_path / "gone.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(sock_path))
    sock.close()
    regular = tmp_path / "keep.txt"
    regular.write_text("keep", encoding="utf-8")

    remove_uds_path(sock_path)
    remove_uds_path(regular)
    remove_uds_path(tmp_path / "does-not-exist.sock")

    assert not sock_path.exists()
    assert regular.exists()


# --------------------------------------------------------------------------
# Server wiring — uvicorn is mocked, so this runs on every platform.
# --------------------------------------------------------------------------


def _bind_kwargs_for(**config_kwargs: object) -> dict[str, object]:
    """Run run_server far enough to capture what it would bind to."""
    from headroom.proxy.server import ProxyConfig, run_server

    captured: dict[str, object] = {}

    def fake_run_uvicorn(  # noqa: ANN202
        app_target,  # noqa: ANN001
        bind_kwargs,  # noqa: ANN001
        workers,  # noqa: ANN001
        limit_concurrency,  # noqa: ANN001
        log_level,  # noqa: ANN001
        uvicorn_kwargs,  # noqa: ANN001
    ):
        captured.update(bind_kwargs)

    with (
        patch("headroom.proxy.server._run_uvicorn", side_effect=fake_run_uvicorn),
        patch("headroom.proxy.server.create_app"),
    ):
        run_server(ProxyConfig(**config_kwargs), print_banner=False)  # type: ignore[arg-type]

    return captured


@requires_core
def test_run_server_binds_host_and_port_by_default() -> None:
    bind = _bind_kwargs_for(host="127.0.0.1", port=9123)

    assert bind == {"host": "127.0.0.1", "port": 9123}


@requires_uds
@requires_core
def test_run_server_binds_the_socket_instead_of_a_port(tmp_path: Path) -> None:
    """uvicorn treats uds and host/port as alternatives; passing both is an error."""
    target = tmp_path / "headroom.sock"

    bind = _bind_kwargs_for(host="127.0.0.1", port=9123, uds=str(target))

    assert bind == {"uds": str(target)}
    assert "host" not in bind and "port" not in bind


@requires_uds
@requires_core
def test_run_server_removes_the_socket_on_exit(tmp_path: Path) -> None:
    """A crash inside uvicorn must not leave an inode that blocks the restart."""
    from headroom.proxy.server import ProxyConfig, run_server

    target = tmp_path / "headroom.sock"

    def bind_then_fail(  # noqa: ANN202
        app_target,  # noqa: ANN001
        bind_kwargs,  # noqa: ANN001
        workers,  # noqa: ANN001
        limit_concurrency,  # noqa: ANN001
        log_level,  # noqa: ANN001
        uvicorn_kwargs,  # noqa: ANN001
    ):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(bind_kwargs["uds"])
        sock.close()
        raise KeyboardInterrupt

    with (
        patch("headroom.proxy.server._run_uvicorn", side_effect=bind_then_fail),
        patch("headroom.proxy.server.create_app"),
        pytest.raises(KeyboardInterrupt),
    ):
        run_server(ProxyConfig(uds=str(target)), print_banner=False)

    assert not target.exists()
