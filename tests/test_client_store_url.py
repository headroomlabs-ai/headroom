"""``HeadroomConfig.store_url`` must decide where ``HeadroomClient`` stores metrics.

The client copied ``config.store_url`` into ``self._store_url`` (which
``validate_setup()`` reports) but opened storage with its own ``store_url``
argument, so metrics went to the temp-directory default while the client
reported the configured path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from headroom.client import HeadroomClient
from headroom.config import HeadroomConfig
from headroom.providers.openai import OpenAIProvider


@pytest.fixture
def temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the client's temp-directory default at an isolated folder."""
    temp = tmp_path / "system-temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return temp


def _client(**kwargs: object) -> HeadroomClient:
    return HeadroomClient(
        original_client=MagicMock(),
        provider=OpenAIProvider(),
        enable_cache_optimizer=False,
        **kwargs,
    )


def _storage_path(client: HeadroomClient) -> Path:
    # "sqlite:////abs/path" parses to "//abs/path"; resolve() normalizes it.
    return Path(client._storage.db_path).resolve()


def test_config_store_url_is_where_metrics_are_stored(tmp_path: Path, temp_dir: Path) -> None:
    db = tmp_path / "mine.db"
    client = _client(config=HeadroomConfig(store_url=f"sqlite:///{db}"))

    assert _storage_path(client) == db.resolve()
    assert db.exists()
    assert not (temp_dir / "headroom.db").exists()
    assert client.validate_setup()["storage"]["url"] == f"sqlite:///{db}"


def test_config_store_url_takes_precedence_over_the_argument(
    tmp_path: Path, temp_dir: Path
) -> None:
    from_config = tmp_path / "from-config.db"
    from_argument = tmp_path / "from-argument.db"
    client = _client(
        store_url=f"sqlite:///{from_argument}",
        config=HeadroomConfig(store_url=f"sqlite:///{from_config}"),
    )

    assert _storage_path(client) == from_config.resolve()
    assert not from_argument.exists()


def test_argument_is_used_when_config_has_no_store_url(tmp_path: Path, temp_dir: Path) -> None:
    from_argument = tmp_path / "from-argument.db"
    config = HeadroomConfig()
    client = _client(store_url=f"sqlite:///{from_argument}", config=config)

    assert _storage_path(client) == from_argument.resolve()
    assert config.store_url is None


def test_shared_config_does_not_leak_one_clients_url_into_another(
    tmp_path: Path, temp_dir: Path
) -> None:
    """One config object reused for two clients, each with its own database."""
    shared = HeadroomConfig()
    a_db = tmp_path / "a.db"
    b_db = tmp_path / "b.db"

    client_a = _client(store_url=f"sqlite:///{a_db}", config=shared)
    client_b = _client(store_url=f"sqlite:///{b_db}", config=shared)

    assert _storage_path(client_a) == a_db.resolve()
    assert _storage_path(client_b) == b_db.resolve()
    assert shared.store_url is None


def test_default_config_stores_in_temp_dir_not_cwd(
    tmp_path: Path, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    client = _client(config=HeadroomConfig())

    assert _storage_path(client) == (temp_dir / "headroom.db").resolve()
    assert not (cwd / "headroom.db").exists()
