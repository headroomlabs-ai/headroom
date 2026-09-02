"""Tests for the wrap_targets.json overlay (registry v2 config system).

Contract under test (see wrap_registry module docstring):
- read-if-exists at ~/.headroom/config/wrap_targets.json, never auto-created
- required version key; unknown version rejects the whole file loudly
- per-field overlay onto code defaults; per-target atomic (no chimera targets)
- precedence unchanged: env vars still beat file values downstream
- new targets are data-only; behavior-crossing fields are override-only
- fail-open: a broken file yields exactly the built-in registry
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from click.testing import CliRunner

import headroom.providers.wrap_registry as wr
from headroom.cli.wrap import wrap


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Point the config root at a tmp dir and reset the resolution cache."""
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path))
    wr._reset_wrap_targets_cache()
    yield tmp_path
    wr._reset_wrap_targets_cache()


def write_config(config_dir, payload) -> None:
    (config_dir / "wrap_targets.json").write_text(json.dumps(payload))
    wr._reset_wrap_targets_cache()


class TestNoConfigFile:
    def test_resolved_registry_is_builtin_object(self, config_dir):
        # Identity, not equality: the no-file path must do zero copying.
        assert wr.resolved_wrap_targets() is wr.WRAP_TARGETS

    def test_status_reports_absent_file(self, config_dir):
        status = wr.wrap_targets_overlay_status()
        assert not status.exists
        assert status.fingerprint is None
        assert status.ok


class TestVersionGate:
    @pytest.mark.parametrize("payload", [{}, {"version": 2, "targets": {}}, {"version": "1"}])
    def test_wrong_or_missing_version_rejects_whole_file(self, config_dir, payload):
        payload.setdefault("targets", {"bob": {"default_mode": "cache"}})
        write_config(config_dir, payload)
        assert wr.resolved_wrap_targets()["bob"].default_mode == "token"
        status = wr.wrap_targets_overlay_status()
        assert status.warnings and "version" in status.warnings[0]

    def test_corrupt_json_falls_back_to_builtins(self, config_dir):
        (config_dir / "wrap_targets.json").write_text("{not json")
        wr._reset_wrap_targets_cache()
        assert wr.resolved_wrap_targets() == wr.WRAP_TARGETS
        assert wr.wrap_targets_overlay_status().warnings


class TestOverlay:
    def test_per_field_override_applies(self, config_dir):
        write_config(config_dir, {"version": 1, "targets": {"bob": {"default_mode": "cache"}}})
        bob = wr.get_wrap_target("bob")
        assert bob.default_mode == "cache"
        # Untouched fields keep code defaults.
        assert bob.openai_api_url == "https://api.us-east.bob.ibm.com/inference/v1"

    def test_mode_alias_normalized_at_load(self, config_dir):
        write_config(
            config_dir, {"version": 1, "targets": {"bob": {"default_mode": "cost_savings"}}}
        )
        assert wr.get_wrap_target("bob").default_mode == "cache"

    def test_per_target_atomic_no_chimera(self, config_dir):
        # One valid field + one invalid field in the same target: the WHOLE
        # target overlay is skipped — never a half-applied chimera.
        write_config(
            config_dir,
            {
                "version": 1,
                "targets": {
                    "bob": {
                        "default_mode": "cache",
                        "origin_passthrough_strip_json_keys": "not-a-list",
                    }
                },
            },
        )
        assert wr.get_wrap_target("bob") == wr.WRAP_TARGETS["bob"]
        outcome = wr.wrap_targets_overlay_status().outcomes[0]
        assert outcome.action == "skipped"

    def test_bad_target_does_not_poison_sibling(self, config_dir):
        write_config(
            config_dir,
            {
                "version": 1,
                "targets": {
                    "bob": {"default_mode": "nonsense-mode"},
                    "goose": {"install_hint": "brew install goose"},
                },
            },
        )
        assert wr.get_wrap_target("bob") == wr.WRAP_TARGETS["bob"]
        assert wr.get_wrap_target("goose").install_hint == "brew install goose"

    def test_unknown_field_skips_target(self, config_dir):
        write_config(config_dir, {"version": 1, "targets": {"bob": {"default_moed": "cache"}}})
        outcome = wr.wrap_targets_overlay_status().outcomes[0]
        assert outcome.action == "skipped"
        assert "unknown field" in outcome.errors[0]

    def test_unhashable_env_var_style_skips_target_not_crash(self, config_dir):
        # A JSON list where a string is expected raised TypeError from the
        # `style in _STYLE_BUILDERS` check, and the validator caught only
        # ValueError -- so a typo in the file crashed every headroom command
        # at import instead of failing open.
        write_config(
            config_dir,
            {
                "version": 1,
                "targets": {"bob": {"env_vars": [{"key": "X", "style": ["openai_v1"]}]}},
            },
        )
        assert wr.get_wrap_target("bob") == wr.WRAP_TARGETS["bob"]
        outcome = wr.wrap_targets_overlay_status().outcomes[0]
        assert outcome.action == "skipped"
        assert "style" in outcome.errors[0]

    def test_empty_binaries_skips_target(self, config_dir):
        # An empty list validated as OK and then `headroom wrap <name>` died
        # with IndexError on binaries[0] instead of the not-found message.
        write_config(config_dir, {"version": 1, "targets": {"bob": {"binaries": []}}})
        assert wr.get_wrap_target("bob") == wr.WRAP_TARGETS["bob"]
        outcome = wr.wrap_targets_overlay_status().outcomes[0]
        assert outcome.action == "skipped"
        assert "binaries" in outcome.errors[0]


class TestNewTargets:
    NEW = {
        "binaries": ["mytool"],
        "install_hint": "pip install mytool",
        "env_vars": [{"key": "OPENAI_BASE_URL", "style": "openai_v1"}],
    }

    def test_data_only_new_target_resolves(self, config_dir):
        write_config(config_dir, {"version": 1, "targets": {"mytool": self.NEW}})
        target = wr.get_wrap_target("mytool")
        assert target.binaries == ("mytool",)
        assert target.env_vars == (wr.EnvVar("OPENAI_BASE_URL", "openai_v1"),)
        assert wr.wrap_targets_overlay_status().outcomes[0].action == "added"

    def test_behavior_crossing_fields_rejected_on_new_targets(self, config_dir):
        entry = dict(self.NEW, extra_chat_routes=["/x/v1/chat/completions"])
        write_config(config_dir, {"version": 1, "targets": {"mytool": entry}})
        assert "mytool" not in wr.resolved_wrap_targets()
        outcome = wr.wrap_targets_overlay_status().outcomes[0]
        assert outcome.action == "skipped"
        assert "proxy routing" in outcome.errors[0]

    def test_new_target_missing_required_fields_skipped(self, config_dir):
        write_config(config_dir, {"version": 1, "targets": {"mytool": {"install_hint": "pip"}}})
        assert "mytool" not in wr.resolved_wrap_targets()


class TestPrecedence:
    def test_env_still_beats_file_default_mode(self, config_dir, monkeypatch):
        """File replaces the CODE DEFAULT only; a live HEADROOM_MODE export wins."""
        import headroom.cli.wrap as wrap_mod

        write_config(config_dir, {"version": 1, "targets": {"bob": {"default_mode": "cache"}}})
        monkeypatch.setenv("HEADROOM_MODE", "token")
        monkeypatch.setattr(wrap_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        captured: dict[str, str | None] = {}

        def fake_launch_tool(**kwargs):
            import os

            captured["mode"] = os.environ.get("HEADROOM_MODE")

        monkeypatch.setattr(wrap_mod, "_launch_tool", fake_launch_tool)
        result = CliRunner().invoke(wrap, ["bob"])
        assert result.exit_code == 0, result.output
        assert captured["mode"] == "token"


class TestDefaultArgs:
    @staticmethod
    def _launch_args(config_dir, monkeypatch, cli_args):
        import headroom.cli.wrap as wrap_mod

        monkeypatch.delenv("HEADROOM_MODE", raising=False)
        monkeypatch.setattr(wrap_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        captured: dict[str, tuple] = {}

        def fake_launch_tool(**kwargs):
            captured["args"] = tuple(kwargs["args"])

        monkeypatch.setattr(wrap_mod, "_launch_tool", fake_launch_tool)
        result = CliRunner().invoke(wrap, cli_args)
        assert result.exit_code == 0, result.output
        return captured["args"]

    def test_config_default_args_prepended_before_invocation_args(self, config_dir, monkeypatch):
        write_config(
            config_dir,
            {"version": 1, "targets": {"bob": {"default_args": ["--auto-approve"]}}},
        )
        args = self._launch_args(config_dir, monkeypatch, ["bob", "run", "hi"])
        assert args == ("--auto-approve", "run", "hi")

    def test_no_default_args_leaves_invocation_args_untouched(self, config_dir, monkeypatch):
        args = self._launch_args(config_dir, monkeypatch, ["bob", "run", "hi"])
        assert args == ("run", "hi")

    def test_invalid_default_args_skips_target(self, config_dir):
        write_config(
            config_dir, {"version": 1, "targets": {"bob": {"default_args": "--auto-approve"}}}
        )
        assert wr.get_wrap_target("bob").default_args == ()
        assert wr.wrap_targets_overlay_status().outcomes[0].action == "skipped"


class TestOriginIndexUsesOverlay:
    def test_overridden_strip_keys_apply(self, config_dir):
        write_config(
            config_dir,
            {
                "version": 1,
                "targets": {
                    "bob": {
                        "origin_passthrough_strip_json_keys": [
                            ["/admin/v1/profile", "region_domain"],
                            ["/admin/v1/profile", "extra_key"],
                        ]
                    }
                },
            },
        )
        body = json.dumps({"region_domain": "x", "extra_key": "y", "keep": 1}).encode()
        out = wr.strip_origin_passthrough_response_keys(
            "https://api.us-east.bob.ibm.com/inference", "/admin/v1/profile", body
        )
        assert out is not None
        assert json.loads(out) == {"keep": 1}

    def test_passthrough_prefix_from_builtin_still_matches(self, config_dir):
        # No config file: index built from code defaults.
        url = wr.resolve_origin_passthrough_url(
            "https://api.us-east.bob.ibm.com/inference", "/inference/v1/model/info"
        )
        assert url == "https://api.us-east.bob.ibm.com/inference/v1/model/info"


class TestFingerprint:
    def test_fingerprint_none_without_file(self, config_dir):
        assert wr.wrap_targets_config_fingerprint() is None
        assert wr.current_wrap_targets_file_fingerprint() is None

    def test_fingerprint_tracks_file_content(self, config_dir):
        write_config(config_dir, {"version": 1, "targets": {}})
        loaded = wr.wrap_targets_config_fingerprint()
        assert loaded and loaded == wr.current_wrap_targets_file_fingerprint()
        # Edit the file after load: on-disk fingerprint moves, loaded stays.
        (config_dir / "wrap_targets.json").write_text(
            json.dumps({"version": 1, "targets": {"bob": {"default_mode": "cache"}}})
        )
        assert wr.current_wrap_targets_file_fingerprint() != loaded
        assert wr.wrap_targets_config_fingerprint() == loaded


class TestDescriptorParity:
    def test_every_dataclass_field_has_a_descriptor(self):
        field_names = {f.name for f in dataclasses.fields(wr.WrapTarget)} - {"name"}
        assert field_names == set(wr._TARGET_FIELDS)

    def test_coercers_produce_dataclass_compatible_values(self):
        # Round-trip bob's own values through the coercers: output must equal
        # the dataclass values, pinning types (tuples, EnvVar) not just names.
        bob = wr.WRAP_TARGETS["bob"]
        assert wr._TARGET_FIELDS["binaries"].coerce(list(bob.binaries)) == bob.binaries
        assert (
            wr._TARGET_FIELDS["env_vars"].coerce(
                [{"key": v.key, "style": v.style, "display": v.display} for v in bob.env_vars]
            )
            == bob.env_vars
        )
        assert (
            wr._TARGET_FIELDS["origin_passthrough_strip_json_keys"].coerce(
                [list(pair) for pair in bob.origin_passthrough_strip_json_keys]
            )
            == bob.origin_passthrough_strip_json_keys
        )
        assert wr._TARGET_FIELDS["default_mode"].coerce(bob.default_mode) == bob.default_mode


class TestProxyReuseWarnings:
    """_warn_wrap_config_staleness: warning-only drift checks on proxy reuse."""

    @staticmethod
    def _warnings(config_dir, monkeypatch, running_config, env_mode=None):
        from headroom.cli.wrap import _warn_wrap_config_staleness

        if env_mode is None:
            monkeypatch.delenv("HEADROOM_MODE", raising=False)
        else:
            monkeypatch.setenv("HEADROOM_MODE", env_mode)
        lines: list[str] = []
        monkeypatch.setattr("click.echo", lines.append)
        _warn_wrap_config_staleness(running_config)
        return lines

    def test_silent_when_everything_matches(self, config_dir, monkeypatch):
        lines = self._warnings(
            config_dir, monkeypatch, {"mode": "cache", "wrap_targets_config_hash": None}
        )
        assert lines == []

    def test_warns_on_mode_mismatch(self, config_dir, monkeypatch):
        lines = self._warnings(
            config_dir,
            monkeypatch,
            {"mode": "cache", "wrap_targets_config_hash": None},
            env_mode="token",
        )
        assert len(lines) == 1 and "'token' mode" in lines[0] and "'cache' mode" in lines[0]

    def test_mode_alias_normalized_before_compare(self, config_dir, monkeypatch):
        # cost_savings is an alias of cache: no mismatch, no warning.
        lines = self._warnings(
            config_dir,
            monkeypatch,
            {"mode": "cache", "wrap_targets_config_hash": None},
            env_mode="cost_savings",
        )
        assert lines == []

    def test_no_mode_warning_without_requested_mode(self, config_dir, monkeypatch):
        lines = self._warnings(
            config_dir, monkeypatch, {"mode": "token", "wrap_targets_config_hash": None}
        )
        assert lines == []

    def test_warns_on_stale_wrap_targets_hash(self, config_dir, monkeypatch):
        write_config(config_dir, {"version": 1, "targets": {}})
        lines = self._warnings(
            config_dir, monkeypatch, {"mode": "cache", "wrap_targets_config_hash": None}
        )
        assert len(lines) == 1 and "wrap_targets.json" in lines[0]

    def test_old_proxy_without_mode_field_is_silent(self, config_dir, monkeypatch):
        # A pre-upgrade proxy's /health has no "mode" key: no false warning.
        lines = self._warnings(
            config_dir,
            monkeypatch,
            {"wrap_targets_config_hash": None},
            env_mode="token",
        )
        assert lines == []


class TestCliSurface:
    """Regression: config presence must not disturb bespoke wrap commands
    (claude, opencode) and must keep registry commands (bob) working."""

    @pytest.mark.parametrize("name", ["claude", "opencode", "bob"])
    def test_wrap_help_unaffected_by_overlay(self, config_dir, name):
        write_config(config_dir, {"version": 1, "targets": {"bob": {"default_mode": "cache"}}})
        result = CliRunner().invoke(wrap, [name, "--help"])
        assert result.exit_code == 0, result.output

    def test_targets_command_reports_and_passes(self, config_dir):
        write_config(config_dir, {"version": 1, "targets": {"bob": {"default_mode": "cache"}}})
        result = CliRunner().invoke(wrap, ["targets"])
        assert result.exit_code == 0, result.output
        assert "overridden: default_mode (mode)" in result.output
        assert "default_mode=cache" in result.output

    def test_targets_command_fails_on_bad_entry(self, config_dir):
        write_config(config_dir, {"version": 1, "targets": {"bob": {"default_mode": "bogus"}}})
        result = CliRunner().invoke(wrap, ["targets"])
        assert result.exit_code == 1
        assert "skipped" in result.output

    def test_targets_command_without_file(self, config_dir):
        result = CliRunner().invoke(wrap, ["targets"])
        assert result.exit_code == 0, result.output
        assert "not present" in result.output
