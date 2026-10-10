"""Tests for the ``headroom.cli_extension`` third-party subcommand seam."""

from __future__ import annotations

import click
import pytest

from headroom.cli import extensions


class _FakeEntry:
    """Stand-in for an ``importlib.metadata.EntryPoint``."""

    def __init__(self, name: str, loaded: object, load_raises: bool = False) -> None:
        self.name = name
        self._loaded = loaded
        self._load_raises = load_raises

    def load(self) -> object:
        if self._load_raises:
            raise ImportError("boom")
        return self._loaded


@pytest.fixture
def group() -> click.Group:
    @click.group()
    def root() -> None:
        pass

    @root.command(name="proxy")
    def proxy() -> None:
        pass

    return root


def _patch_entries(monkeypatch: pytest.MonkeyPatch, entries: list[_FakeEntry]) -> None:
    monkeypatch.setattr(
        extensions.importlib.metadata,
        "entry_points",
        lambda group: entries,
    )


def test_registers_new_command(monkeypatch: pytest.MonkeyPatch, group: click.Group) -> None:
    @click.command(name="econ")
    def econ() -> None:
        pass

    _patch_entries(monkeypatch, [_FakeEntry("fleet", lambda main: main.add_command(econ))])

    assert extensions.register_all(group) == ["fleet"]
    assert "econ" in group.commands
    assert "proxy" in group.commands


def test_raising_registrant_is_skipped(monkeypatch: pytest.MonkeyPatch, group: click.Group) -> None:
    def bad(main: click.Group) -> None:
        raise RuntimeError("unlicensed")

    _patch_entries(monkeypatch, [_FakeEntry("broken", bad)])

    assert extensions.register_all(group) == []
    assert set(group.commands) == {"proxy"}


def test_partial_registration_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch, group: click.Group
) -> None:
    """A registrant that adds a command then raises leaves nothing behind."""

    @click.command(name="half")
    def half() -> None:
        pass

    def bad(main: click.Group) -> None:
        main.add_command(half)
        raise RuntimeError("failed after partial work")

    _patch_entries(monkeypatch, [_FakeEntry("broken", bad)])

    assert extensions.register_all(group) == []
    assert "half" not in group.commands


def test_load_failure_is_skipped(monkeypatch: pytest.MonkeyPatch, group: click.Group) -> None:
    _patch_entries(monkeypatch, [_FakeEntry("stale", None, load_raises=True)])

    assert extensions.register_all(group) == []
    assert set(group.commands) == {"proxy"}


def test_cannot_shadow_builtin_command(monkeypatch: pytest.MonkeyPatch, group: click.Group) -> None:
    """Overriding a built-in IS a silent behavior change, so it is refused."""
    builtin = group.commands["proxy"]

    @click.command(name="proxy")
    def evil_proxy() -> None:
        pass

    _patch_entries(monkeypatch, [_FakeEntry("evil", lambda main: main.add_command(evil_proxy))])

    assert extensions.register_all(group) == []
    assert group.commands["proxy"] is builtin


def test_shadowing_registrant_does_not_block_others(
    monkeypatch: pytest.MonkeyPatch, group: click.Group
) -> None:
    @click.command(name="proxy")
    def evil_proxy() -> None:
        pass

    @click.command(name="econ")
    def econ() -> None:
        pass

    _patch_entries(
        monkeypatch,
        [
            _FakeEntry("evil", lambda main: main.add_command(evil_proxy)),
            _FakeEntry("fleet", lambda main: main.add_command(econ)),
        ],
    )

    assert extensions.register_all(group) == ["fleet"]
    assert "econ" in group.commands


def test_enumeration_failure_is_survivable(
    monkeypatch: pytest.MonkeyPatch, group: click.Group
) -> None:
    def boom(group: str) -> list[_FakeEntry]:
        raise RuntimeError("no metadata")

    monkeypatch.setattr(extensions.importlib.metadata, "entry_points", boom)

    assert extensions.register_all(group) == []
    assert set(group.commands) == {"proxy"}
