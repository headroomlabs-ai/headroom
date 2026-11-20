"""``project_key_strategy_from_config`` (GH #4712 helper plumbing)."""

from __future__ import annotations

from types import SimpleNamespace

from headroom.memory.storage_router import ProjectKeyStrategy
from headroom.proxy.helpers import project_key_strategy_from_config


def test_none_for_missing_or_legacy_config() -> None:
    # Config objects predating the field (SimpleNamespace, mocks) must
    # yield None, meaning the resolver default (``cwd``), never raise.
    assert project_key_strategy_from_config(None) is None
    assert project_key_strategy_from_config(SimpleNamespace()) is None
    assert project_key_strategy_from_config(object()) is None


def test_none_for_default_cwd_value() -> None:
    config = SimpleNamespace(memory_project_key_strategy="cwd")
    assert project_key_strategy_from_config(config) is None


def test_none_for_empty_value() -> None:
    config = SimpleNamespace(memory_project_key_strategy="")
    assert project_key_strategy_from_config(config) is None


def test_git_for_git_value() -> None:
    config = SimpleNamespace(memory_project_key_strategy="git")
    strategy = project_key_strategy_from_config(config)
    assert strategy == ProjectKeyStrategy(mode="git")


def test_normalizes_case_and_whitespace() -> None:
    config = SimpleNamespace(memory_project_key_strategy="  GIT ")
    strategy = project_key_strategy_from_config(config)
    assert strategy == ProjectKeyStrategy(mode="git")
