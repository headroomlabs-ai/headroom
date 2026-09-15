"""HEADROOM_TOIN_BACKEND=none must mean in-memory-only, not "use the default".

`_create_default_toin_backend` reports its decision only through a return value,
and it returned `None` both for "nothing configured" and for the explicit
`none` setting. `ToolIntelligenceNetwork.__init__` reads `None` as "no explicit
backend" and falls back to `FileSystemTOINBackend(config.storage_path)`, and the
default storage_path is never empty -- so `none` kept writing toin.json.
"""

from __future__ import annotations

import json

from headroom.telemetry.toin import (
    TOIN_BACKEND_ENV_VAR,
    TOINConfig,
    ToolIntelligenceNetwork,
    toin_backend_disabled,
)


def _network(tmp_path, name: str = "toin.json") -> tuple[ToolIntelligenceNetwork, object]:
    path = tmp_path / name
    return ToolIntelligenceNetwork(TOINConfig(storage_path=str(path))), path


def test_backend_none_keeps_toin_in_memory(tmp_path, monkeypatch):
    """The regression: a non-empty storage_path must not defeat `none`."""
    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "none")

    toin, path = _network(tmp_path)

    assert toin._backend is None
    toin.save()
    assert not path.exists()
    assert not list(tmp_path.rglob("*.json"))


def test_backend_none_is_case_and_whitespace_insensitive(tmp_path, monkeypatch):
    """Matches how the value is normalized for every other backend name."""
    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "  NONE  ")

    toin, path = _network(tmp_path)

    assert toin._backend is None
    toin.save()
    assert not path.exists()


def test_backend_none_still_records_patterns_in_memory(tmp_path, monkeypatch):
    """In-memory-only disables persistence, not learning."""
    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "none")
    from headroom.telemetry.models import ToolSignature

    toin, path = _network(tmp_path)
    toin.record_compression(
        tool_signature=ToolSignature(
            structure_hash="abc123",
            field_count=2,
            has_nested_objects=False,
            has_arrays=True,
            max_depth=1,
            string_field_count=1,
            has_error_like_field=False,
            has_message_like_field=True,
        ),
        original_count=100,
        compressed_count=10,
        original_tokens=1000,
        compressed_tokens=100,
        strategy="test",
    )

    assert toin.get_stats()["patterns_tracked"] == 1
    assert not path.exists()


def test_unset_backend_still_persists(tmp_path, monkeypatch):
    """Control: the default path must keep writing, or the fix broke persistence."""
    monkeypatch.delenv(TOIN_BACKEND_ENV_VAR, raising=False)

    toin, path = _network(tmp_path)

    assert toin._backend is not None
    toin.save()
    assert path.exists()
    assert json.loads(path.read_text())["version"] == "2.0"


def test_filesystem_backend_still_persists(tmp_path, monkeypatch):
    """Control: the explicit filesystem name is unaffected."""
    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "filesystem")

    toin, path = _network(tmp_path)

    assert toin._backend is not None
    toin.save()
    assert path.exists()


def test_explicit_backend_argument_beats_the_env(tmp_path, monkeypatch):
    """A caller-supplied backend still wins -- `none` must not veto injection."""
    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "none")

    saved: list[dict] = []

    class _RecordingBackend:
        def load(self) -> dict:
            return {}

        def save(self, data: dict) -> None:
            saved.append(data)

    path = tmp_path / "toin.json"
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path)), backend=_RecordingBackend())

    toin.save()
    assert len(saved) == 1
    assert not path.exists()


def test_toin_backend_disabled_predicate(monkeypatch):
    monkeypatch.delenv(TOIN_BACKEND_ENV_VAR, raising=False)
    assert toin_backend_disabled() is False

    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "none")
    assert toin_backend_disabled() is True

    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "filesystem")
    assert toin_backend_disabled() is False

    monkeypatch.setenv(TOIN_BACKEND_ENV_VAR, "redis")
    assert toin_backend_disabled() is False
