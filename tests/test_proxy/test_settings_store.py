"""Tests for the file-backed HEADROOM_* settings store (Phase 1).

Covers: JSON round-trip with coercion, unknown-key drop, fail-open load on a
corrupt file, atomic save, setdefault precedence (explicit export wins), the
effective-value resolution order, and secret masking in the schema/GET views.
"""

import json
import os
from dataclasses import replace

import pytest

from headroom import paths, settings_store


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the workspace dir (and thus settings.json) at an isolated tmp dir."""
    monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path))
    # Ensure no per-resource override leaks in from the ambient environment.
    monkeypatch.delenv(settings_store.paths.HEADROOM_SETTINGS_PATH_ENV, raising=False)
    return tmp_path


def _clear_env(monkeypatch):
    for field in settings_store.SETTINGS:
        monkeypatch.delenv(field.env, raising=False)


class TestRoundTrip:
    def test_save_then_load_coerces_types(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        settings_store.save(
            {
                "port": "9898",  # str in → int out
                "target_ratio": "0.4",  # str in → float out
                "disable_kompress": "1",  # str in → bool out
                "savings_profile": "balanced",
            }
        )
        loaded = settings_store.load()
        assert loaded == {
            "port": 9898,
            "target_ratio": 0.4,
            "disable_kompress": True,
            "savings_profile": "balanced",
        }

    def test_load_drops_unknown_keys(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        path = paths.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"port": 9000, "bogus_key": 1}', encoding="utf-8")
        assert settings_store.load() == {"port": 9000}

    def test_save_is_atomic_no_temp_left_behind(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        settings_store.save({"port": 9000})
        leftovers = [p.name for p in workspace.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []
        assert paths.settings_path().exists()


class TestValidation:
    def test_save_rejects_unknown_key(self, workspace):
        with pytest.raises(settings_store.SettingsValidationError) as exc:
            settings_store.save({"nope": 1})
        assert exc.value.unknown_keys == ["nope"]

    def test_save_rejects_out_of_range(self, workspace):
        with pytest.raises(settings_store.SettingsValidationError) as exc:
            settings_store.save({"target_ratio": 5.0})
        assert "target_ratio" in exc.value.field_errors

    def test_save_rejects_bad_enum(self, workspace):
        with pytest.raises(settings_store.SettingsValidationError) as exc:
            settings_store.save({"savings_profile": "nonsense"})
        assert "savings_profile" in exc.value.field_errors

    def test_save_rejects_non_numeric(self, workspace):
        with pytest.raises(settings_store.SettingsValidationError) as exc:
            settings_store.save({"rpm": "abc"})
        assert "rpm" in exc.value.field_errors

    def test_save_rejects_non_finite_float(self, workspace):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(settings_store.SettingsValidationError) as exc:
                settings_store.save({"budget": bad})
            assert "budget" in exc.value.field_errors

    def test_optional_bool_empty_string_coerces_to_none(self, workspace):
        # Empty optional-bool means "inherit" -> dropped, not forced to False.
        assert settings_store.validate({"disable_kompress_anthropic": ""}) == {}
        # A plain bool empty string still coerces to False.
        assert settings_store.validate({"disable_kompress": ""}) == {"disable_kompress": False}


class TestFailOpen:
    def test_corrupt_json_returns_empty(self, workspace):
        path = paths.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        assert settings_store.load() == {}

    def test_non_object_json_returns_empty(self, workspace):
        path = paths.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert settings_store.load() == {}

    def test_missing_file_returns_empty(self, workspace):
        assert settings_store.load() == {}


class TestApplyToEnviron:
    def test_setdefault_fills_unset_env(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        settings_store.apply_to_environ({"port": 9898, "disable_kompress": True})
        assert os.environ["HEADROOM_PORT"] == "9898"
        assert os.environ["HEADROOM_DISABLE_KOMPRESS"] == "1"

    def test_explicit_export_wins(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HEADROOM_PORT", "7777")
        settings_store.apply_to_environ({"port": 9898})
        assert os.environ["HEADROOM_PORT"] == "7777"

    def test_bool_false_serializes_to_zero(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        settings_store.apply_to_environ({"code_aware_enabled": False})
        assert os.environ["HEADROOM_CODE_AWARE_ENABLED"] == "0"


class TestEffectiveValues:
    def test_default_then_file_then_env(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        # default
        assert settings_store.effective_values()["savings_profile"] == "coding"
        # file overrides default
        settings_store.save({"savings_profile": "balanced"})
        assert settings_store.effective_values()["savings_profile"] == "balanced"
        # env overrides file
        monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", "general")
        assert settings_store.effective_values()["savings_profile"] == "general"


class TestSecretMasking:
    def test_schema_and_stored_mask_secret(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        # No curated knob is secret; synthesize one to exercise the masking path.
        base_field = next(f for f in settings_store.SETTINGS if f.key == "log_file")
        secret_field = replace(base_field, secret=True)
        registry = tuple(
            secret_field if f.key == "log_file" else f for f in settings_store.SETTINGS
        )
        monkeypatch.setattr(settings_store, "SETTINGS", registry)
        monkeypatch.setattr(settings_store, "_BY_KEY", {f.key: f for f in registry})
        settings_store.save({"log_file": "/tmp/secret.log"})

        stored = settings_store.stored_values()
        assert stored["log_file"] == settings_store._MASK

        schema = settings_store.to_schema()
        log_field = next(f for f in schema["fields"] if f["key"] == "log_file")
        assert log_field["value"] == settings_store._MASK
        assert log_field["stored"] == settings_store._MASK
        # unmasked read still returns the real value for internal callers
        assert settings_store.stored_values(mask_secrets=False)["log_file"] == "/tmp/secret.log"

    def test_schema_lists_all_keys_as_restart_required(self, workspace, monkeypatch):
        _clear_env(monkeypatch)
        schema = settings_store.to_schema()
        assert schema["needs_restart_keys"] == [f.key for f in settings_store.SETTINGS]
        assert "Compression" in schema["groups"]

    def test_anthropic_extra_headers_retain_on_mask(self, workspace, monkeypatch):
        """Saving _MASK for anthropic_extra_headers retains the stored value."""
        _clear_env(monkeypatch)
        settings_store.save({"anthropic_extra_headers": '{"Api-Key": "secret123"}'})
        stored = settings_store.load()
        assert stored.get("anthropic_extra_headers") == '{"Api-Key": "secret123"}'

        settings_store.save({"anthropic_extra_headers": settings_store._MASK})
        stored = settings_store.load()
        assert stored.get("anthropic_extra_headers") == '{"Api-Key": "secret123"}', (
            "Saving _MASK should retain the stored value, not overwrite it"
        )

    def test_anthropic_extra_headers_clear_on_none(self, workspace, monkeypatch):
        """Saving None for anthropic_extra_headers removes it."""
        _clear_env(monkeypatch)
        settings_store.save({"anthropic_extra_headers": '{"Api-Key": "secret123"}'})
        stored = settings_store.load()
        assert "anthropic_extra_headers" in stored

        settings_store.save({"anthropic_extra_headers": None})
        stored = settings_store.load()
        assert "anthropic_extra_headers" not in stored

    def test_anthropic_extra_headers_json_validation_roundtrip(self, workspace, monkeypatch):
        """JSON header-map values are canonical and round-trip correctly."""
        _clear_env(monkeypatch)
        settings_store.save({"anthropic_extra_headers": '{"Z-Header": "val", "A-Header": "val2"}'})
        stored = settings_store.load()
        canonical = json.dumps({"A-Header": "val2", "Z-Header": "val"}, sort_keys=True)
        assert stored.get("anthropic_extra_headers") == canonical

    def test_anthropic_extra_headers_invalid_json_raises(self, workspace, monkeypatch):
        """Invalid JSON in anthropic_extra_headers field raises SettingsValidationError."""
        _clear_env(monkeypatch)
        from headroom.settings_store import SettingsValidationError

        with pytest.raises(SettingsValidationError) as exc_info:
            settings_store.save({"anthropic_extra_headers": "not json"})
        assert "anthropic_extra_headers" in exc_info.value.field_errors

    def test_anthropic_extra_headers_non_string_values_raises(self, workspace, monkeypatch):
        """Header-map JSON with non-string values raises SettingsValidationError."""
        _clear_env(monkeypatch)
        from headroom.settings_store import SettingsValidationError

        with pytest.raises(SettingsValidationError) as exc_info:
            settings_store.save({"anthropic_extra_headers": '{"header": 123}'})
        assert "anthropic_extra_headers" in exc_info.value.field_errors

    def test_anthropic_extra_headers_non_object_raises(self, workspace, monkeypatch):
        """Header-map with non-object JSON raises SettingsValidationError."""
        _clear_env(monkeypatch)
        from headroom.settings_store import SettingsValidationError

        with pytest.raises(SettingsValidationError) as exc_info:
            settings_store.save({"anthropic_extra_headers": '["header", "value"]'})
        assert "anthropic_extra_headers" in exc_info.value.field_errors

    def test_openai_extra_headers_retain_on_mask(self, workspace, monkeypatch):
        """Saving _MASK for openai_extra_headers retains the stored value."""
        _clear_env(monkeypatch)
        settings_store.save({"openai_extra_headers": '{"Authorization": "Bearer token"}'})
        stored = settings_store.load()
        assert stored.get("openai_extra_headers") == '{"Authorization": "Bearer token"}'

        settings_store.save({"openai_extra_headers": settings_store._MASK})
        stored = settings_store.load()
        assert stored.get("openai_extra_headers") == '{"Authorization": "Bearer token"}'

    def test_anthropic_base_url_plain_str_field(self, workspace, monkeypatch):
        """anthropic_base_url behaves as a plain str field."""
        _clear_env(monkeypatch)
        settings_store.save({"anthropic_base_url": "https://custom.example.com/v1"})
        stored = settings_store.load()
        assert stored.get("anthropic_base_url") == "https://custom.example.com/v1"

        settings_store.save({"anthropic_base_url": "https://other.example.com"})
        stored = settings_store.load()
        assert stored.get("anthropic_base_url") == "https://other.example.com"

    def test_openai_base_url_plain_str_field(self, workspace, monkeypatch):
        """openai_base_url behaves as a plain str field."""
        _clear_env(monkeypatch)
        settings_store.save({"openai_base_url": "https://custom.openai.example.com/v1"})
        stored = settings_store.load()
        assert stored.get("openai_base_url") == "https://custom.openai.example.com/v1"


class TestRegistryDriftAgainstClick:
    """Guards against the registry silently falling behind the real CLI surface.

    Introspects the ``proxy`` Click command's own parameter objects (not a
    regex over the source) so a future contributor who adds a new
    ``@click.option(..., envvar="HEADROOM_...")`` to cli/proxy.py without
    adding a matching SettingField gets a failing test, not a silent gap.
    """

    def test_every_headroom_click_envvar_is_in_the_registry(self):
        from headroom.cli.proxy import proxy

        click_envvars = {
            param.envvar
            for param in proxy.params
            if isinstance(getattr(param, "envvar", None), str)
            and param.envvar.startswith("HEADROOM_")
        }
        registry_envvars = {field.env for field in settings_store.SETTINGS}
        missing = click_envvars - registry_envvars
        assert not missing, (
            f"New HEADROOM_* Click option(s) not covered by settings_store.SETTINGS: "
            f"{sorted(missing)}. Add a SettingField for each, or document why it's "
            "deliberately excluded (e.g. a secret)."
        )
