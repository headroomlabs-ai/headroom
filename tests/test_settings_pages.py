"""Page taxonomy, live-knob metadata, and helper coverage for settings_store.

Covers the category-page data model: every field maps to a known sidebar page,
``to_schema`` emits ``page``/``live``/``pages``, and the live-knob helpers
(``live_keys``, ``runtime_overrides``, ``coerce_env_value``) behave as the
``/settings`` handlers rely on.
"""

from __future__ import annotations

import os

import pytest

from headroom import settings_store
from headroom.settings_store import PAGES, SETTINGS

_LIVE_KEYS = {
    "output_shaper",
    "verbosity_level",
    "effort_router",
    "mechanical_effort",
    "verbosity_autotune",
    "output_holdout",
    "intercept_read_min_chars",
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolate settings.json under a tmp workspace and clear ambient env knobs."""
    monkeypatch.setenv(settings_store.paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(settings_store.paths.HEADROOM_SETTINGS_PATH_ENV, raising=False)
    for field in SETTINGS:
        monkeypatch.delenv(field.env, raising=False)
    return tmp_path


class TestPageTaxonomy:
    def test_every_field_has_a_known_page(self):
        for field in SETTINGS:
            assert field.page in PAGES, f"{field.key} -> {field.page!r} not in PAGES"

    def test_schema_pages_populated_and_ordered(self, workspace):
        schema = settings_store.to_schema()
        populated = [p for p in PAGES if any(f.page == p for f in SETTINGS)]
        assert schema["pages"] == populated
        # The remap fills every page, so all nine appear in nav order.
        assert schema["pages"] == list(PAGES)

    def test_no_ccr_moved_to_ccr_and_caching(self):
        assert settings_store._BY_KEY["no_ccr"].page == "CCR & Caching"
        assert settings_store._BY_KEY["no_ccr_proactive_expansion"].page == "CCR & Caching"

    def test_schema_fields_carry_page_and_live(self, workspace):
        by_key = {f["key"]: f for f in settings_store.to_schema()["fields"]}
        assert by_key["output_shaper"]["page"] == "Output Shaping"
        assert by_key["output_shaper"]["live"] is True
        assert by_key["target_ratio"]["live"] is False

    def test_added_curated_knobs_on_expected_pages(self):
        expected = {
            "kompress_backend": "Compression",
            "mode": "Compression",
            "dedupe": "Compression",
            "tool_search": "Compression",
            "ccr_backend": "CCR & Caching",
            "redis_url": "CCR & Caching",
            "ccr_ttl_seconds": "CCR & Caching",
            "stateless": "Networking & Security",
            "offline": "Networking & Security",
            "tls_strict": "Networking & Security",
            "cors_origins": "Networking & Security",
            "ws_origins": "Networking & Security",
            "vertex_base_url": "Endpoints",
            "bedrock_base_url": "Endpoints",
            "gemini_base_url": "Endpoints",
            "cloudcode_base_url": "Endpoints",
            "qdrant_url": "Memory",
            "qdrant_host": "Memory",
            "qdrant_port": "Memory",
            "qdrant_api_key": "Memory",
            "optimize": "Compression",
            "intercept_enabled": "Compression",
            "cache_enabled": "CCR & Caching",
            "rate_limit_enabled": "Limits & Budget",
            "embedding_server": "Memory",
            "embedding_server_socket": "Memory",
        }
        for key, page in expected.items():
            field = settings_store._BY_KEY[key]
            assert field.page == page, f"{key} -> {field.page!r}, expected {page!r}"
            assert field.live is False, f"{key} should be restart-required, not live"
        assert settings_store._BY_KEY["qdrant_api_key"].secret is True


class TestLiveKnobs:
    def test_only_output_shaping_is_live(self):
        assert {f.key for f in SETTINGS if f.live} == _LIVE_KEYS
        assert all(f.page == "Output Shaping" for f in SETTINGS if f.live)

    def test_live_keys_helper_filters_non_live(self):
        assert settings_store.live_keys(["output_shaper", "target_ratio", "no_ccr"]) == [
            "output_shaper"
        ]

    def test_runtime_overrides_serializes_live_only(self):
        overrides = settings_store.runtime_overrides(
            ["output_shaper", "verbosity_level", "target_ratio"],
            {"output_shaper": True, "verbosity_level": 3, "target_ratio": 0.5},
        )
        assert overrides == {"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3"}

    def test_runtime_overrides_skips_absent_keys(self):
        assert settings_store.runtime_overrides(["verbosity_level"], {}) == {}

    def test_coerce_env_value_roundtrips(self):
        assert settings_store.coerce_env_value("verbosity_level", "3") == 3
        assert settings_store.coerce_env_value("output_shaper", "on") is True
        assert settings_store.coerce_env_value("verbosity_level", "") is None
        assert settings_store.coerce_env_value("verbosity_level", None) is None
        assert settings_store.coerce_env_value("verbosity_level", "notanint") is None
        assert settings_store.coerce_env_value("nope", "1") is None


class TestObservabilityKnobs:
    @pytest.mark.parametrize(
        "key",
        [
            "otel_metrics_enabled",
            "otel_metrics_endpoint",
            "otel_service_name",
            "langfuse_enabled",
            "telemetry",
            "periodic_toin_stats",
        ],
    )
    def test_registered_on_observability_page(self, key):
        assert settings_store._BY_KEY[key].page == "Observability"

    def test_optional_bool_telemetry_roundtrip(self, workspace):
        settings_store.save({"telemetry": True})
        assert settings_store.load()["telemetry"] is True
        settings_store.apply_to_environ(settings_store.load())
        # "1" is one of the beacon's accepted on-values.
        assert os.environ["HEADROOM_TELEMETRY"] == "1"


class TestLivePersistence:
    def test_apply_to_environ_skips_live_knobs(self, workspace):
        settings_store.apply_to_environ({"output_shaper": True, "target_ratio": 0.5})
        # Non-live knobs seed the environment; live knobs do not (they use the
        # runtime_env override store, so they stay GUI-editable after a restart).
        assert os.environ["HEADROOM_TARGET_RATIO"] == "0.5"
        assert "HEADROOM_OUTPUT_SHAPER" not in os.environ

    def test_env_for(self):
        assert settings_store.env_for("output_shaper") == "HEADROOM_OUTPUT_SHAPER"
        assert settings_store.env_for("nope") is None


class TestCuratedKnobBehavior:
    def test_bool_knob_applies_to_environ_as_one(self, workspace):
        settings_store.save({"stateless": True})
        settings_store.apply_to_environ(settings_store.load())
        assert os.environ["HEADROOM_STATELESS"] == "1"

    def test_tls_strict_defaults_true_and_serializes_off(self, workspace):
        assert settings_store._BY_KEY["tls_strict"].default is True
        # Relaxing it must serialize to "0", one of the reader's off-values.
        settings_store.save({"tls_strict": False})
        settings_store.apply_to_environ(settings_store.load())
        assert os.environ["HEADROOM_TLS_STRICT"] == "0"

    def test_qdrant_port_range_enforced(self, workspace):
        with pytest.raises(settings_store.SettingsValidationError):
            settings_store.save({"qdrant_port": 70000})
        settings_store.save({"qdrant_port": 6333})
        assert settings_store.load()["qdrant_port"] == 6333

    def test_kompress_backend_enum_rejects_unknown(self, workspace):
        with pytest.raises(settings_store.SettingsValidationError):
            settings_store.save({"kompress_backend": "cuda"})
        settings_store.save({"kompress_backend": "pytorch_mps"})
        assert settings_store.load()["kompress_backend"] == "pytorch_mps"

    def test_qdrant_api_key_masked_in_schema(self, workspace):
        settings_store.save({"qdrant_api_key": "qdr-secret-123"})
        by_key = {f["key"]: f for f in settings_store.to_schema()["fields"]}
        assert by_key["qdrant_api_key"]["value"] == settings_store._MASK
        # Round-trips: resending the mask retains the stored secret.
        settings_store.save({"qdrant_api_key": settings_store._MASK})
        assert settings_store.load()["qdrant_api_key"] == "qdr-secret-123"


class TestProxyModeKnob:
    def test_mode_is_a_basic_compression_knob(self):
        field = settings_store._BY_KEY["mode"]
        assert field.env == "HEADROOM_MODE"
        assert field.default == "cache"
        assert field.page == "Compression"
        assert field.tier == "basic"
        assert field.live is False

    def test_mode_enum_roundtrip_and_apply(self, workspace):
        settings_store.save({"mode": "cache"})
        assert settings_store.load()["mode"] == "cache"
        settings_store.apply_to_environ(settings_store.load())
        assert os.environ["HEADROOM_MODE"] == "cache"

    def test_mode_rejects_unknown_value(self, workspace):
        with pytest.raises(settings_store.SettingsValidationError):
            settings_store.save({"mode": "turbo"})


class TestPrefixCacheUnlockKnobs:
    """Both knobs govern whether a warm provider prefix may be recompressed."""

    @pytest.mark.parametrize(
        ("key", "env"),
        [
            ("cold_recompact", "HEADROOM_COLD_RECOMPACT"),
            ("net_cost_policy", "HEADROOM_NET_COST_POLICY"),
        ],
    )
    def test_advanced_compression_bool_defaulting_off(self, key, env):
        field = settings_store._BY_KEY[key]
        assert field.env == env
        assert field.type == "bool"
        assert field.default is False
        assert field.page == "Compression"
        assert field.tier == "advanced"
        assert field.live is False

    @pytest.mark.parametrize("key", ["cold_recompact", "net_cost_policy"])
    def test_enabling_serializes_to_one(self, workspace, key):
        # Both readers gate on the literal "1": content_router compares
        # `== "1"`, the cold-prefix hook accepts "1"/"true"/"yes".
        settings_store.save({key: True})
        settings_store.apply_to_environ(settings_store.load())
        assert os.environ[settings_store.env_for(key)] == "1"


class TestCliArgToggles:
    def test_positive_toggles_default_enabled(self):
        for key in ("optimize", "cache_enabled", "rate_limit_enabled"):
            assert settings_store._BY_KEY[key].default is True, key

    def test_disable_toggle_serializes_zero(self, workspace):
        settings_store.save({"optimize": False, "cache_enabled": False})
        settings_store.apply_to_environ(settings_store.load())
        assert os.environ["HEADROOM_OPTIMIZE"] == "0"
        assert os.environ["HEADROOM_CACHE_ENABLED"] == "0"

    def test_intercept_and_embedding_server_are_bools(self):
        assert settings_store._BY_KEY["intercept_enabled"].type == "bool"
        assert settings_store._BY_KEY["intercept_enabled"].default is False
        assert settings_store._BY_KEY["embedding_server"].default is False

    def test_intercept_enabled_applies_as_one(self, workspace):
        settings_store.save({"intercept_enabled": True})
        settings_store.apply_to_environ(settings_store.load())
        assert os.environ["HEADROOM_INTERCEPT_ENABLED"] == "1"

    def test_embedding_server_socket_str_roundtrip(self, workspace):
        settings_store.save({"embedding_server_socket": "/tmp/e.sock"})
        assert settings_store.load()["embedding_server_socket"] == "/tmp/e.sock"


class TestSettingsWinPrecedence:
    def test_stored_value_overrides_env_and_is_not_locked(self, workspace, monkeypatch):
        # settings.json is highest priority for a normal knob.
        monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", "general")
        settings_store.save({"savings_profile": "balanced"})
        by_key = {f["key"]: f for f in settings_store.to_schema()["fields"]}
        sp = by_key["savings_profile"]
        assert sp["value"] == "balanced"  # file wins in the effective value
        assert sp["env_override"] is False  # a normal knob is not env-locked
        assert sp["env_present"] is True  # UI can still note the env var is set

    def test_manifest_managed_env_still_wins_and_locks(self, workspace, monkeypatch):
        # manifest_managed knobs keep env/manifest precedence and stay locked.
        monkeypatch.setenv("HEADROOM_PORT", "7777")
        settings_store.save({"port": 9898})
        by_key = {f["key"]: f for f in settings_store.to_schema()["fields"]}
        port = by_key["port"]
        assert port["value"] == 7777
        assert port["env_override"] is True
