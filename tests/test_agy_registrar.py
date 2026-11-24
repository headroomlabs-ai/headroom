"""Tests for headroom.mcp_registry.agy.AgyRegistrar.

All tests use a tmp_path home_dir seam so the real ~/.gemini is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.mcp_registry.agy import AgyRegistrar
from headroom.mcp_registry.base import RegisterStatus, ServerSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = ServerSpec(
    name="headroom",
    command="headroom",
    args=("mcp", "serve"),
    env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
)

_OTHER_SPEC = ServerSpec(
    name="other-server",
    command="/usr/bin/other",
    args=("--flag",),
    env={},
)


def _make_reg(tmp_path: Path) -> AgyRegistrar:
    return AgyRegistrar(home_dir=tmp_path)


def _config_path(tmp_path: Path) -> Path:
    # agy 1.1.x read-path (migrated from .gemini/antigravity-cli/mcp_config.json).
    return tmp_path / ".gemini" / "config" / "mcp_config.json"


def _write_config(tmp_path: Path, data: dict) -> None:
    p = _config_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def _read_config(tmp_path: Path) -> dict:
    p = _config_path(tmp_path)
    return json.loads(p.read_text())


def test_config_writes_use_lf_newlines(tmp_path: Path) -> None:
    reg = _make_reg(tmp_path)
    reg.register_server(_SPEC)
    data = _config_path(tmp_path).read_bytes()
    assert b"\n" in data
    assert b"\r\n" not in data


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


class TestDetect:
    def test_returns_false_when_dir_absent(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert reg.detect() is False

    def test_returns_true_when_appdata_dir_exists(self, tmp_path: Path) -> None:
        # detect() keys on agy's app-data dir (the stable install marker), not
        # the migrated config dir (which may not exist until the first write).
        (tmp_path / ".gemini" / "antigravity-cli").mkdir(parents=True, exist_ok=True)
        reg = _make_reg(tmp_path)
        assert reg.detect() is True

    def test_returns_true_when_config_file_exists(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"mcpServers": {}})
        reg = _make_reg(tmp_path)
        assert reg.detect() is True


# ---------------------------------------------------------------------------
# get_server
# ---------------------------------------------------------------------------


class TestGetServer:
    def test_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert reg.get_server("headroom") is None

    def test_returns_none_when_server_absent(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"mcpServers": {}})
        reg = _make_reg(tmp_path)
        assert reg.get_server("headroom") is None

    def test_returns_spec_when_present(self, tmp_path: Path) -> None:
        _write_config(
            tmp_path,
            {
                "mcpServers": {
                    "headroom": {
                        "command": "headroom",
                        "args": ["mcp", "serve"],
                        "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
                    }
                }
            },
        )
        reg = _make_reg(tmp_path)
        spec = reg.get_server("headroom")
        assert spec is not None
        assert spec.name == "headroom"
        assert spec.command == "headroom"
        assert tuple(spec.args) == ("mcp", "serve")
        assert spec.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"}

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        p = _config_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json")
        reg = _make_reg(tmp_path)
        assert reg.get_server("headroom") is None


# ---------------------------------------------------------------------------
# register_server — REGISTERED
# ---------------------------------------------------------------------------


class TestRegisterServer:
    def test_malformed_config_is_not_overwritten(self, tmp_path: Path) -> None:
        """A config we cannot parse must abort the write, not get replaced.

        ``mcp_config.json`` is shared with the Antigravity IDE and holds the
        user's own servers; treating an unreadable file as ``{}`` and writing our
        single entry back would delete all of them.
        """
        p = _config_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        original = '{"mcpServers": {"user-server": {"command": "x"}}, TRUNCATED'
        p.write_text(original)

        result = _make_reg(tmp_path).register_server(_SPEC)

        assert result.status == RegisterStatus.FAILED
        assert p.read_text() == original

    def test_unregister_leaves_malformed_config_untouched(self, tmp_path: Path) -> None:
        p = _config_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        original = "{not valid json"
        p.write_text(original)

        assert _make_reg(tmp_path).unregister_server("headroom") is False
        assert p.read_text() == original

    def test_registers_new_server(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        assert "headroom" in config["mcpServers"]
        entry = config["mcpServers"]["headroom"]
        assert entry["command"] == "headroom"
        assert entry["args"] == ["mcp", "serve"]

    def test_registers_creates_parent_dirs(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert not _config_path(tmp_path).parent.exists()
        reg.register_server(_SPEC)
        assert _config_path(tmp_path).exists()

    # ALREADY
    def test_already_when_spec_matches(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.ALREADY

    # MISMATCH without force
    def test_mismatch_when_command_differs_no_force(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        different = ServerSpec(
            name="headroom",
            command="/different/path/headroom",
            args=("mcp", "serve"),
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
        )
        result = reg.register_server(different)
        assert result.status == RegisterStatus.MISMATCH
        # Config must be unchanged
        config = _read_config(tmp_path)
        assert config["mcpServers"]["headroom"]["command"] == "headroom"

    def test_mismatch_when_env_differs_no_force(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        different = ServerSpec(
            name="headroom",
            command="headroom",
            args=("mcp", "serve"),
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:1111"},
        )
        result = reg.register_server(different)
        assert result.status == RegisterStatus.MISMATCH

    # force overwrite
    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        updated = ServerSpec(
            name="headroom",
            command="/new/headroom",
            args=("mcp", "serve"),
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
        )
        result = reg.register_server(updated, force=True)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        assert config["mcpServers"]["headroom"]["command"] == "/new/headroom"

    # MERGE: other entries preserved
    def test_merge_preserves_other_user_servers(self, tmp_path: Path) -> None:
        # Pre-populate with a user-managed server.
        _write_config(
            tmp_path,
            {
                "mcpServers": {
                    "user-server": {
                        "command": "/usr/bin/user-mcp",
                        "args": ["--some-flag"],
                    }
                }
            },
        )
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        # Both entries must exist.
        assert "user-server" in config["mcpServers"]
        assert "headroom" in config["mcpServers"]
        # User entry untouched.
        assert config["mcpServers"]["user-server"]["command"] == "/usr/bin/user-mcp"

    def test_missing_file_treated_as_empty(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.REGISTERED

    def test_registers_spec_without_env(self, tmp_path: Path) -> None:
        spec = ServerSpec(name="minimal", command="headroom", args=())
        reg = _make_reg(tmp_path)
        result = reg.register_server(spec)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        entry = config["mcpServers"]["minimal"]
        # env key should not be present when empty.
        assert "env" not in entry


# ---------------------------------------------------------------------------
# unregister_server
# ---------------------------------------------------------------------------


class TestUnregisterServer:
    def test_returns_false_when_file_absent(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert reg.unregister_server("headroom") is False

    def test_returns_false_when_server_absent(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"mcpServers": {}})
        reg = _make_reg(tmp_path)
        assert reg.unregister_server("headroom") is False

    def test_removes_named_server(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        removed = reg.unregister_server("headroom")
        assert removed is True
        config = _read_config(tmp_path)
        assert "headroom" not in config["mcpServers"]

    def test_unregister_only_removes_named_server(self, tmp_path: Path) -> None:
        """Unregistering 'headroom' MUST NOT remove other user entries."""
        _write_config(
            tmp_path,
            {
                "mcpServers": {
                    "user-server": {"command": "/bin/user-mcp"},
                    "headroom": {
                        "command": "headroom",
                        "args": ["mcp", "serve"],
                    },
                }
            },
        )
        reg = _make_reg(tmp_path)
        reg.unregister_server("headroom")
        config = _read_config(tmp_path)
        assert "user-server" in config["mcpServers"]
        assert "headroom" not in config["mcpServers"]

    def test_idempotent_double_unregister(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        assert reg.unregister_server("headroom") is True
        assert reg.unregister_server("headroom") is False


class TestRegistrarSafety:
    def test_conflict_detail_redacts_env_values(self, tmp_path: Path) -> None:
        """Conflict messages must never echo user env values (potential credentials)."""
        secret = "sk-user-super-secret-token-999"
        _write_config(
            tmp_path,
            {
                "mcpServers": {
                    "headroom": {
                        "command": "headroom",
                        "args": ["mcp", "serve"],
                        "env": {"API_TOKEN": secret},
                    }
                }
            },
        )
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC, force=False)
        assert result.status == RegisterStatus.MISMATCH
        assert "API_TOKEN" in result.detail
        assert secret not in result.detail

    def test_failed_write_preserves_existing_config_and_cleans_temp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write error mid-serialization must leave the shared mcp_config.json intact."""
        import headroom.mcp_registry.agy as agy_mod

        original = {
            "mcpServers": {
                "ide-server": {"command": "/usr/local/bin/ide-mcp"},
            }
        }
        _write_config(tmp_path, original)
        config_path = tmp_path / ".gemini" / "config" / "mcp_config.json"
        before_bytes = config_path.read_bytes()

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("simulated disk full")

        monkeypatch.setattr(agy_mod.json, "dump", _boom)

        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.FAILED
        assert config_path.read_bytes() == before_bytes
        leftover_tmps = list(config_path.parent.glob("*.tmp"))
        assert leftover_tmps == []

    @pytest.mark.windows_newline
    def test_write_json_uses_lf_newlines(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.REGISTERED
        raw = (tmp_path / ".gemini" / "config" / "mcp_config.json").read_bytes()
        assert b"\n" in raw
        assert b"\r\n" not in raw

    def test_concurrent_registrations_preserve_both_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two concurrent registrations for different servers must both survive."""
        import threading

        import headroom.mcp_registry.agy as agy_mod

        _write_config(
            tmp_path,
            {"mcpServers": {"existing": {"command": "/bin/existing"}}},
        )
        reg = _make_reg(tmp_path)
        spec_a = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        spec_b = ServerSpec(name="serena", command="uvx", args=("serena", "start-mcp-server"))

        entered = threading.Event()
        release = threading.Event()
        orig_write_json = agy_mod._write_json
        call_count = 0
        count_lock = threading.Lock()

        def _coordinated_write(path: Path, data: dict) -> None:
            nonlocal call_count
            with count_lock:
                call_count += 1
                is_first = call_count == 1
            if is_first:
                entered.set()
                # Hold the first writer briefly so an unlocked second caller
                # would read the stale pre-write config and clobber this write.
                release.wait(timeout=0.3)
            orig_write_json(path, data)

        monkeypatch.setattr(agy_mod, "_write_json", _coordinated_write)

        results: dict[str, object] = {}

        def _worker(label: str, spec: ServerSpec) -> None:
            results[label] = reg.register_server(spec)

        t1 = threading.Thread(target=_worker, args=("a", spec_a))
        t2 = threading.Thread(target=_worker, args=("b", spec_b))
        t1.start()
        assert entered.wait(timeout=5.0)
        t2.start()
        # Give t2 a moment to contend on the lock (or race past read if unlocked).
        t2.join(timeout=0.1)
        release.set()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive() and not t2.is_alive()
        assert results["a"].status == RegisterStatus.REGISTERED  # type: ignore[attr-defined]
        assert results["b"].status == RegisterStatus.REGISTERED  # type: ignore[attr-defined]
        config = _read_config(tmp_path)
        assert set(config["mcpServers"].keys()) == {"existing", "headroom", "serena"}

    def test_cross_process_concurrent_registrations_preserve_both_entries(
        self, tmp_path: Path
    ) -> None:
        """Two separate Headroom processes registering different servers must both survive."""
        import subprocess
        import sys

        _write_config(
            tmp_path,
            {"mcpServers": {"existing": {"command": "/bin/existing"}}},
        )
        worker_script = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "import headroom.mcp_registry.agy as agy_mod\n"
            "from headroom.mcp_registry.base import ServerSpec\n"
            "home = Path(sys.argv[1])\n"
            "name = sys.argv[2]\n"
            "delay = float(sys.argv[3])\n"
            "orig_write = agy_mod._write_json\n"
            "def slow_write(path, data):\n"
            "    if delay:\n"
            "        time.sleep(delay)\n"
            "    orig_write(path, data)\n"
            "agy_mod._write_json = slow_write\n"
            "reg = agy_mod.AgyRegistrar(home_dir=home)\n"
            "res = reg.register_server(ServerSpec(name=name, command='cmd-' + name, args=()))\n"
            "print(res.status.value)\n"
        )

        p1 = subprocess.Popen(
            [sys.executable, "-c", worker_script, str(tmp_path), "headroom", "0.25"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        p2 = subprocess.Popen(
            [sys.executable, "-c", worker_script, str(tmp_path), "serena", "0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out1, err1 = p1.communicate(timeout=15)
        out2, err2 = p2.communicate(timeout=15)
        assert p1.returncode == 0, err1
        assert p2.returncode == 0, err2
        assert out1.strip() == RegisterStatus.REGISTERED.value
        assert out2.strip() == RegisterStatus.REGISTERED.value

        config = _read_config(tmp_path)
        assert set(config["mcpServers"].keys()) == {"existing", "headroom", "serena"}

    def test_agy_and_antigravity_registrars_share_lock_and_preserve_both_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AgyRegistrar and AntigravityRegistrar must serialize access to shared mcp_config.json."""
        import threading

        import headroom.mcp_registry.agy as agy_mod
        from headroom.mcp_registry.antigravity import AntigravityRegistrar

        _write_config(
            tmp_path,
            {"mcpServers": {"existing": {"command": "/bin/existing"}}},
        )
        agy_reg = _make_reg(tmp_path)
        ide_reg = AntigravityRegistrar(home_dir=tmp_path)
        cli_spec = ServerSpec(name="headroom-cli", command="headroom", args=("mcp", "serve"))
        ide_spec = ServerSpec(name="headroom-ide", command="headroom", args=("mcp", "serve"))

        entered = threading.Event()
        release = threading.Event()
        orig_write_json = agy_mod._write_json

        def _paused_agy_write(path: Path, data: dict) -> None:
            entered.set()
            release.wait(timeout=0.3)
            orig_write_json(path, data)

        monkeypatch.setattr(agy_mod, "_write_json", _paused_agy_write)

        results: dict[str, object] = {}

        def _run_agy() -> None:
            results["agy"] = agy_reg.register_server(cli_spec)

        def _run_ide() -> None:
            results["ide"] = ide_reg.register_server(ide_spec)

        t1 = threading.Thread(target=_run_agy)
        t2 = threading.Thread(target=_run_ide)
        t1.start()
        assert entered.wait(timeout=5.0)
        t2.start()
        t2.join(timeout=0.1)
        release.set()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive() and not t2.is_alive()
        assert results["agy"].status == RegisterStatus.REGISTERED  # type: ignore[attr-defined]
        assert results["ide"].status == RegisterStatus.REGISTERED  # type: ignore[attr-defined]
        config = _read_config(tmp_path)
        assert set(config["mcpServers"].keys()) == {"existing", "headroom-cli", "headroom-ide"}

    def test_windows_config_file_lock_raises_permanent_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Permanent msvcrt.locking errors must fail immediately and propagate to the registrar."""
        import errno
        import sys
        import types

        import headroom.mcp_registry.base as base_mod

        calls = 0

        def _fake_locking(fd: int, mode: int, nbytes: int) -> None:
            nonlocal calls
            calls += 1
            if calls > 3:
                raise AssertionError("config_file_lock retried indefinitely on permanent EBADF")
            raise OSError(errno.EBADF, "Bad file descriptor")

        fake_msvcrt = types.SimpleNamespace(
            LK_NBLCK=2,
            LK_UNLCK=0,
            locking=_fake_locking,
        )
        monkeypatch.setattr(base_mod.sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
        monkeypatch.setattr(base_mod.time, "sleep", lambda _secs: None)

        config_path = tmp_path / ".gemini" / "antigravity" / "mcp_config.json"
        with pytest.raises(OSError) as exc_info:
            with base_mod.config_file_lock(config_path):
                pass
        assert exc_info.value.errno == errno.EBADF
        assert calls == 1

        # Registrar should receive the error cleanly and report FAILED (thread lock released).
        reg = _make_reg(tmp_path)
        calls = 0
        result = reg.register_server(
            ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        )
        assert result.status == RegisterStatus.FAILED
        assert calls == 1

    def test_windows_config_file_lock_retries_contended_lock_until_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contended msvcrt.locking calls retry until the lock is granted, then unlock on exit."""
        import errno
        import sys
        import types

        import headroom.mcp_registry.base as base_mod

        events: list[int] = []
        sleeps: list[float] = []

        def _fake_locking(fd: int, mode: int, nbytes: int) -> None:
            events.append(mode)
            if mode == 2 and len(events) <= 2:
                raise PermissionError(errno.EACCES, "Permission denied")

        fake_msvcrt = types.SimpleNamespace(
            LK_NBLCK=2,
            LK_UNLCK=0,
            locking=_fake_locking,
        )
        monkeypatch.setattr(base_mod.sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
        monkeypatch.setattr(base_mod.time, "sleep", lambda secs: sleeps.append(secs))

        config_path = tmp_path / ".gemini" / "antigravity" / "mcp_config.json"
        entered = False
        with base_mod.config_file_lock(config_path):
            entered = True

        assert entered is True
        assert events == [2, 2, 2, 0]
        assert len(sleeps) == 2

    def test_windows_config_file_lock_does_not_write_empty_lockfile_before_locking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows locking must lock byte 0 past EOF without writing a sentinel byte first."""
        import os
        import sys
        import types

        import headroom.mcp_registry.base as base_mod

        seen_sizes: list[int] = []

        def _fake_locking(fd: int, mode: int, nbytes: int) -> None:
            if mode == 2:
                seen_sizes.append(os.fstat(fd).st_size)

        fake_msvcrt = types.SimpleNamespace(
            LK_NBLCK=2,
            LK_UNLCK=0,
            locking=_fake_locking,
        )
        monkeypatch.setattr(base_mod.sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

        config_path = _config_path(tmp_path)
        with base_mod.config_file_lock(config_path):
            pass

        assert seen_sizes == [0]

    def test_register_and_unregister_preserve_symlinked_mcp_config(self, tmp_path: Path) -> None:
        """Atomic writes must update the symlink target and keep mcp_config.json a symlink."""
        dotfiles = tmp_path / "dotfiles"
        dotfiles.mkdir()
        target = dotfiles / "agy-mcp.json"
        target.write_text(
            json.dumps({"mcpServers": {"existing": {"command": "/bin/existing"}}}, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        config_link = _config_path(tmp_path)
        config_link.parent.mkdir(parents=True, exist_ok=True)
        try:
            config_link.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable on this platform/user: {exc}")

        reg = _make_reg(tmp_path)
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        result = reg.register_server(spec)
        assert result.status == RegisterStatus.REGISTERED
        assert config_link.is_symlink()
        assert config_link.resolve() == target.resolve()
        target_data = json.loads(target.read_text(encoding="utf-8"))
        assert set(target_data["mcpServers"].keys()) == {"existing", "headroom"}

        assert reg.unregister_server("headroom") is True
        assert config_link.is_symlink()
        assert config_link.resolve() == target.resolve()
        target_data = json.loads(target.read_text(encoding="utf-8"))
        assert set(target_data["mcpServers"].keys()) == {"existing"}

    def test_mcp_status_with_null_mcp_servers_reports_unconfigured_and_preserves_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """{"mcpServers": null} must report AGY unconfigured without raising or editing the file."""
        from click.testing import CliRunner

        from headroom.cli.mcp import mcp_status

        cfg = _config_path(tmp_path)
        cfg.parent.mkdir(parents=True, exist_ok=True)
        original_content = '{"mcpServers": null}\n'
        cfg.write_text(original_content, encoding="utf-8", newline="\n")

        reg = _make_reg(tmp_path)
        assert reg.get_server("headroom") is None

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        result = CliRunner().invoke(mcp_status, [])
        assert result.exit_code == 0, result.output
        assert "AttributeError" not in result.output
        assert cfg.read_text(encoding="utf-8") == original_content
