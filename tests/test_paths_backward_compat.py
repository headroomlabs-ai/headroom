"""Adversarial backward-compatibility stress tests for ``headroom.paths``.

These three scenarios codify the cross-cutting guarantees the filesystem
contract must honor so that issue-175 stays strictly additive:

1. A legacy-only user (only ``HEADROOM_SAVINGS_PATH`` set) must keep getting
   their legacy path, byte-for-byte, with the new canonical vars unset.
2. A canonical-only user (only ``HEADROOM_WORKSPACE_DIR`` set) must see every
   workspace-bucket resource relocate under the new root with the correct
   filenames.
3. A user who set both a legacy per-resource env var *and* the new canonical
   env var must see the legacy var win for that resource (precedence:
   explicit > legacy > canonical > default).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom import paths

CANONICAL_ENV_VARS = (
    paths.HEADROOM_CONFIG_DIR_ENV,
    paths.HEADROOM_WORKSPACE_DIR_ENV,
)
LEGACY_ENV_VARS = (
    paths.HEADROOM_SAVINGS_PATH_ENV,
    paths.HEADROOM_TOIN_PATH_ENV,
    paths.HEADROOM_SUBSCRIPTION_STATE_PATH_ENV,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in CANONICAL_ENV_VARS + LEGACY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def fake_home(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    clean_env.setenv("HOME", str(tmp_path))
    clean_env.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def test_legacy_only_user_savings_unchanged(
    fake_home: Path, clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario 1: legacy-only user keeps byte-for-byte legacy semantics.

    Only ``HEADROOM_SAVINGS_PATH`` is set. The canonical workspace/config
    vars are unset. The helper must return the exact legacy value as
    supplied.
    """

    legacy_value = str(tmp_path / "oldstyle" / "savings.json")
    clean_env.setenv(paths.HEADROOM_SAVINGS_PATH_ENV, legacy_value)

    # Byte-for-byte equality (after Path-roundtrip) — no silent rewriting.
    result = paths.savings_path()
    assert result == Path(legacy_value)
    assert str(result) == legacy_value

    # The rest of the world is unaffected: defaults still flow through home.
    assert paths.workspace_dir() == fake_home / ".headroom"
    assert paths.config_dir() == fake_home / ".headroom" / "config"


def test_canonical_only_user_workspace_bucket_relocates(
    fake_home: Path, clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario 2: canonical-only user sees every workspace resource move."""

    alt_ws = tmp_path / "mnt" / "alt"
    clean_env.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(alt_ws))

    # Root + every workspace-bucket helper relocates with the correct name.
    assert paths.workspace_dir() == alt_ws
    # Config derives from workspace when config env unset.
    assert paths.config_dir() == alt_ws / "config"

    assert paths.savings_path() == alt_ws / "proxy_savings.json"
    assert paths.toin_path() == alt_ws / "toin.json"
    assert paths.subscription_state_path() == alt_ws / "subscription_state.json"
    assert paths.memory_db_path() == alt_ws / "memory.db"
    assert paths.native_memory_dir() == alt_ws / "memories"
    assert paths.license_cache_path() == alt_ws / "license_cache.json"
    assert paths.session_stats_path() == alt_ws / "session_stats.jsonl"
    assert paths.sync_state_path() == alt_ws / "sync_state.json"
    assert paths.bridge_state_path() == alt_ws / "bridge_state.json"
    assert paths.log_dir() == alt_ws / "logs"
    assert paths.proxy_log_path() == alt_ws / "logs" / "proxy.log"
    assert paths.debug_400_dir() == alt_ws / "logs" / "debug_400"
    assert paths.bin_dir() == alt_ws / "bin"
    assert paths.proxy_clients_dir(8787) == alt_ws / "clients" / "8787"
    assert paths.deploy_root() == alt_ws / "deploy"
    assert paths.beacon_lock_path(8787) == alt_ws / ".beacon_lock_8787"
    # Config bucket follows the derived config dir.
    assert paths.models_config_path() == alt_ws / "config" / "models.json"


def test_both_set_legacy_wins_over_canonical(
    fake_home: Path, clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario 3: legacy per-resource env wins over canonical root env.

    When both ``HEADROOM_SAVINGS_PATH=/old/...`` and
    ``HEADROOM_WORKSPACE_DIR=/new/...`` are set, ``savings_path()`` must
    return the legacy value. This is the core backward-compat guarantee:
    adding the canonical env var never quietly steals a user's existing
    override.
    """

    legacy = tmp_path / "old" / "savings.json"
    new_ws = tmp_path / "new" / "ws"
    clean_env.setenv(paths.HEADROOM_SAVINGS_PATH_ENV, str(legacy))
    clean_env.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(new_ws))

    # Savings legacy wins.
    assert paths.savings_path() == legacy
    # The other workspace helpers (no legacy var set) relocate under the
    # canonical root — proves orthogonality: one override does not bleed
    # into another.
    assert paths.memory_db_path() == new_ws / "memory.db"
    assert paths.log_dir() == new_ws / "logs"


def test_all_three_legacy_vars_win_simultaneously(
    fake_home: Path, clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression guard: every legacy env var remains honored when canonical
    and legacy are both set for the same resource."""

    savings_legacy = tmp_path / "sv.json"
    toin_legacy = tmp_path / "tn.json"
    sub_legacy = tmp_path / "sb.json"
    new_ws = tmp_path / "ws"

    clean_env.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(new_ws))
    clean_env.setenv(paths.HEADROOM_SAVINGS_PATH_ENV, str(savings_legacy))
    clean_env.setenv(paths.HEADROOM_TOIN_PATH_ENV, str(toin_legacy))
    clean_env.setenv(paths.HEADROOM_SUBSCRIPTION_STATE_PATH_ENV, str(sub_legacy))

    assert paths.savings_path() == savings_legacy
    assert paths.toin_path() == toin_legacy
    assert paths.subscription_state_path() == sub_legacy
