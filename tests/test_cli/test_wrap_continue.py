"""Tests for `headroom wrap continue` command (PR-G1, Phase G)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_inject_continue_rtk_systemmessage_new_file(tmp_path: Path) -> None:
    """Writing into a non-existent config.json creates parents + sets systemMessage."""
    config_file = tmp_path / ".continue" / "config.json"
    assert not config_file.exists()

    assert wrap_mod._inject_continue_rtk_systemmessage(config_file) is True

    data = json.loads(config_file.read_text())
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_inject_continue_rtk_systemmessage_preserves_existing_keys(tmp_path: Path) -> None:
    """Pre-existing keys are not touched; per-model entries get systemMessage."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"models": [{"title": "GPT-4o", "provider": "openai"}]}))

    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    # Pre-existing fields on the model entry are preserved verbatim.
    assert data["models"][0]["title"] == "GPT-4o"
    assert data["models"][0]["provider"] == "openai"
    # Top-level systemMessage is set.
    assert wrap_mod._RTK_MARKER in data["systemMessage"]
    # Per-model systemMessage is also populated (Continue overrides top-level
    # with per-model when set, so we must visit each model).
    assert wrap_mod._RTK_MARKER in data["models"][0]["systemMessage"]


def test_inject_continue_rtk_systemmessage_appends_to_existing_message(
    tmp_path: Path,
) -> None:
    """Pre-existing systemMessage content is preserved; rtk block is appended."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    existing_msg = "You are a helpful assistant."
    config_file.write_text(json.dumps({"systemMessage": existing_msg}))

    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    assert data["systemMessage"].startswith(existing_msg)
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_inject_continue_rtk_systemmessage_idempotent(tmp_path: Path) -> None:
    """Re-injection must not duplicate the marker."""
    config_file = tmp_path / ".continue" / "config.json"

    wrap_mod._inject_continue_rtk_systemmessage(config_file)
    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    assert data["systemMessage"].count(wrap_mod._RTK_MARKER) == 1


def test_inject_continue_rtk_systemmessage_refuses_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON must be left untouched and the helper must return False."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    malformed = '{ "models": [ this is not valid json'
    config_file.write_text(malformed)

    result = wrap_mod._inject_continue_rtk_systemmessage(config_file)

    assert result is False
    assert config_file.read_text() == malformed


def test_inject_continue_rtk_systemmessage_refuses_non_object_root(
    tmp_path: Path,
) -> None:
    """A JSON array at the root is not a valid Continue config; leave untouched."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("[]")

    result = wrap_mod._inject_continue_rtk_systemmessage(config_file)

    assert result is False
    assert config_file.read_text() == "[]"


def test_wrap_continue_prepare_only_injects_systemmessage(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wrap continue --prepare-only` injects into ./.continue/config.json by default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(main, ["wrap", "continue", "--prepare-only"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".continue" / "config.json"
    assert config_file.exists()
    data = json.loads(config_file.read_text())
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_wrap_continue_respects_custom_config_path(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--config writes to the user-specified path, not the cwd default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    custom_config = tmp_path / "custom" / "my-continue.json"

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(
            main,
            ["wrap", "continue", "--prepare-only", "--config", str(custom_config)],
        )

    assert result.exit_code == 0, result.output
    assert custom_config.exists()
    assert not (tmp_path / ".continue" / "config.json").exists()
    data = json.loads(custom_config.read_text())
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


# ---------------------------------------------------------------------------
# H1: non-string systemMessage must NOT be silently clobbered.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "non_string_value",
    [
        {"role": "system", "content": "You are helpful."},  # dict
        ["You are helpful.", "Respond in JSON."],  # list
        42,  # int
    ],
    ids=["dict", "list", "int"],
)
def test_inject_continue_rtk_systemmessage_refuses_non_string_top_level(
    tmp_path: Path,
    non_string_value: object,
) -> None:
    """A non-string top-level systemMessage must NOT be overwritten."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    original_payload = {"systemMessage": non_string_value, "other": "untouched"}
    config_file.write_text(json.dumps(original_payload))
    original_bytes = config_file.read_bytes()

    result = wrap_mod._inject_continue_rtk_systemmessage(config_file)

    assert result is False, "must report refusal when user data would be clobbered"
    # File must be byte-identical to before.
    assert config_file.read_bytes() == original_bytes
    data = json.loads(config_file.read_text())
    assert data["systemMessage"] == non_string_value
    assert data["other"] == "untouched"


@pytest.mark.parametrize(
    "non_string_value",
    [
        {"role": "system", "content": "Per-model system."},
        ["List", "of", "strings"],
        7,
    ],
    ids=["dict", "list", "int"],
)
def test_inject_continue_rtk_systemmessage_refuses_non_string_per_model(
    tmp_path: Path,
    non_string_value: object,
) -> None:
    """A non-string per-model systemMessage must NOT be overwritten."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    original_payload = {
        "models": [
            {"title": "GPT-4o", "provider": "openai", "systemMessage": non_string_value},
        ],
    }
    config_file.write_text(json.dumps(original_payload))

    result = wrap_mod._inject_continue_rtk_systemmessage(config_file)

    assert result is False, "must report refusal when per-model user data would be clobbered"
    data = json.loads(config_file.read_text())
    # The non-string per-model value must be preserved.
    assert data["models"][0]["systemMessage"] == non_string_value


# ---------------------------------------------------------------------------
# M2: per-model systemMessage handling.
# ---------------------------------------------------------------------------


def test_inject_continue_rtk_systemmessage_visits_each_model(
    tmp_path: Path,
) -> None:
    """Each models[i].systemMessage gets the RTK block."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        json.dumps(
            {
                "models": [
                    {"title": "A", "systemMessage": "user value"},
                    {"title": "B"},  # no systemMessage yet
                ],
            }
        )
    )

    assert wrap_mod._inject_continue_rtk_systemmessage(config_file) is True

    data = json.loads(config_file.read_text())
    # Pre-existing per-model systemMessage is preserved + RTK block appended.
    assert "user value" in data["models"][0]["systemMessage"]
    assert wrap_mod._RTK_MARKER in data["models"][0]["systemMessage"]
    # Model with no systemMessage gets the RTK block fresh.
    assert wrap_mod._RTK_MARKER in data["models"][1]["systemMessage"]
    # Top-level also populated.
    assert wrap_mod._RTK_MARKER in data["systemMessage"]


def test_inject_continue_rtk_systemmessage_per_model_idempotent(
    tmp_path: Path,
) -> None:
    """Re-running must not duplicate per-model RTK blocks."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        json.dumps({"models": [{"title": "A", "systemMessage": "user"}]}),
    )

    wrap_mod._inject_continue_rtk_systemmessage(config_file)
    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    assert data["models"][0]["systemMessage"].count(wrap_mod._RTK_MARKER) == 1
    assert data["systemMessage"].count(wrap_mod._RTK_MARKER) == 1


def test_inject_continue_rtk_systemmessage_skips_non_dict_model_entries(
    tmp_path: Path,
) -> None:
    """A models[] entry that isn't a dict must be left untouched."""
    config_file = tmp_path / ".continue" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        json.dumps({"models": ["just a string entry", {"title": "B"}]}),
    )

    wrap_mod._inject_continue_rtk_systemmessage(config_file)

    data = json.loads(config_file.read_text())
    # Non-dict entry preserved verbatim.
    assert data["models"][0] == "just a string entry"
    # Dict entry got the RTK block.
    assert wrap_mod._RTK_MARKER in data["models"][1]["systemMessage"]


# ---------------------------------------------------------------------------
# M4: Ctrl-C during prelude emits a clear message.
# ---------------------------------------------------------------------------


def test_wrap_continue_keyboardinterrupt_during_prelude_emits_clear_message(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C between marker injection and proxy start must signal clearly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    config_file = tmp_path / ".continue" / "config.json"

    def raise_kbd_after_inject(*args, **kwargs):  # noqa: ANN002, ANN003
        # Simulate the user hitting Ctrl-C right after we wrote config.json
        # but before the proxy started.
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text('{"systemMessage": "marker block"}')
        raise KeyboardInterrupt

    with patch.object(wrap_mod, "_ensure_rtk_binary", side_effect=raise_kbd_after_inject):
        result = runner.invoke(main, ["wrap", "continue", "--prepare-only"])

    assert result.exit_code == 130
    assert "interrupted" in result.output.lower()
    assert "idempotent" in result.output.lower()
    assert config_file.exists()
    assert str(config_file) in result.output
