"""The available-models block must inform the agent without harming user content.

Background: `wrap copilot` uses the CLI's BYOK transport, which is single-model.
Native Copilot knows the full model set and picks correctly; under BYOK the agent
has nothing authoritative to consult, so asked to "use a different model" it
invents names or scrapes them from repo files. Observed live from a wrapped
session: it reported `claude-fable-5`, `claude-opus-4.8-fast` and `grok-4.5` as
available -- none are Copilot models -- mixed with real ones.

These tests pin the two properties that make the fix safe to ship: the list is
accurate/refreshable, and the user's own instructions are never damaged.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from headroom.cli.wrap import (
    _copilot_models_instructions_block,
    _inject_copilot_models_instructions,
    _live_copilot_model_ids,
    _remove_copilot_models_instructions,
)

USER_CONTENT = "# My project rules\n\nAlways use tabs. Never edit generated files.\n"


@pytest.fixture
def instructions(tmp_path: Path) -> Path:
    path = tmp_path / ".github" / "copilot-instructions.md"
    path.parent.mkdir(parents=True)
    path.write_text(USER_CONTENT, encoding="utf-8")
    return path


def test_block_lists_ids_and_points_at_the_live_command() -> None:
    block = _copilot_models_instructions_block(["gpt-5.4", "claude-opus-5"])
    assert "`gpt-5.4`" in block
    assert "`claude-opus-5`" in block
    # The pointer matters as much as the list: the block is written at launch,
    # so this is what keeps the answer current mid-session.
    assert "headroom models" in block
    # It must actively discourage the two observed failure modes.
    assert "do not scrape" in block.lower()
    assert "invent" in block.lower()


def test_injection_preserves_user_content(instructions: Path) -> None:
    _inject_copilot_models_instructions(instructions, ["gpt-5.4"])
    text = instructions.read_text(encoding="utf-8")
    assert USER_CONTENT.strip() in text
    assert "gpt-5.4" in text


def test_relaunch_refreshes_the_list_instead_of_duplicating(instructions: Path) -> None:
    """A stale list is the bug being fixed, so the block must update in place."""
    _inject_copilot_models_instructions(instructions, ["gpt-5.4", "gemini-2.5-pro"])
    _inject_copilot_models_instructions(instructions, ["gpt-5.5", "claude-opus-5"])
    text = instructions.read_text(encoding="utf-8")
    assert text.count("<!-- headroom:available-models -->") == 1
    assert "gemini-2.5-pro" not in text  # retired entry gone on refresh
    assert "gpt-5.4" not in text
    assert "gpt-5.5" in text
    assert USER_CONTENT.strip() in text


def test_removal_leaves_only_user_content(instructions: Path) -> None:
    _inject_copilot_models_instructions(instructions, ["gpt-5.4"])
    assert _remove_copilot_models_instructions(instructions) is True
    text = instructions.read_text(encoding="utf-8")
    assert "headroom:available-models" not in text
    assert "gpt-5.4" not in text
    assert USER_CONTENT.strip() in text


def test_removal_is_a_noop_when_absent(instructions: Path) -> None:
    assert _remove_copilot_models_instructions(instructions) is False
    assert instructions.read_text(encoding="utf-8") == USER_CONTENT


def test_creates_the_file_when_missing(tmp_path: Path) -> None:
    path = tmp_path / ".github" / "copilot-instructions.md"
    _inject_copilot_models_instructions(path, ["gpt-5.4"])
    assert path.exists()
    assert "gpt-5.4" in path.read_text(encoding="utf-8")


def test_empty_model_list_writes_nothing(instructions: Path) -> None:
    """Discovery failed => say nothing rather than assert an empty catalogue."""
    assert _inject_copilot_models_instructions(instructions, []) is False
    assert instructions.read_text(encoding="utf-8") == USER_CONTENT


def test_live_ids_only_include_selectable_chat_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    fixture = (
        Path(__file__).parent / "fixtures" / "copilot_models" / "models_list.json"
    ).read_text(encoding="utf-8")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json=json.loads(fixture)))
    ids = _live_copilot_model_ids("https://api.githubcopilot.com", "tok")
    assert "claude-opus-4.8" in ids
    assert "mai-code-1-flash-picker" in ids
    # Not selectable / not chat models must never be advertised to the agent.
    assert "text-embedding-3-small" not in ids
    assert "trajectory-compaction" not in ids
    assert "gpt-4o" not in ids


def test_live_ids_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: object, **k: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert _live_copilot_model_ids("https://api.githubcopilot.com", "tok") == []
    assert _live_copilot_model_ids(None, None) == []


# ---------------------------------------------------------------------------
# Malformed markers and line endings — the cases that let real data loss through
# ---------------------------------------------------------------------------


SECURITY = "## IMPORTANT SECURITY RULES\nNever commit secrets. Never disable the audit log.\n"


def test_user_typed_start_marker_does_not_destroy_their_content(tmp_path: Path) -> None:
    """The data-loss bug: `index(END)` matched the FIRST end marker in the file.

    A user documenting this feature has the start marker in their own prose.
    Launch 1 appended a block; launch 2 then spliced from *their* marker to *our*
    block's end marker and deleted everything between — silently, while printing
    "Refreshed available-models list". Reproduced before the fix:
    `"Never commit secrets" in text` was False.
    """
    p = tmp_path / ".github" / "copilot-instructions.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        f"# Team rules\n\n<!-- headroom:available-models -->\n(documented on purpose)\n\n{SECURITY}",
        encoding="utf-8",
    )
    _inject_copilot_models_instructions(p, ["gpt-5.4"])
    _inject_copilot_models_instructions(p, ["gpt-5.5"])
    text = p.read_text(encoding="utf-8")
    assert "Never commit secrets" in text
    assert "IMPORTANT SECURITY RULES" in text
    assert "# Team rules" in text

    _remove_copilot_models_instructions(p)
    assert "Never commit secrets" in p.read_text(encoding="utf-8")


def test_reversed_markers_do_not_duplicate_user_content(tmp_path: Path) -> None:
    """END before START gave `end < start`, which duplicated user text."""
    p = tmp_path / ".github" / "copilot-instructions.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "<!-- /headroom:available-models -->\nUSER-KEEP\n<!-- headroom:available-models -->\n",
        encoding="utf-8",
    )
    for _ in range(3):
        _inject_copilot_models_instructions(p, ["gpt-5.4"])
    assert p.read_text(encoding="utf-8").count("USER-KEEP") == 1


def test_crlf_line_endings_are_preserved(tmp_path: Path) -> None:
    """Rewriting one marked region must not convert the whole file's endings.

    `_read_text` normalizes CRLF to LF, so a naive rewrite turned every line of a
    source-controlled file into LF and showed up as a whole-file diff on Windows.
    """
    p = tmp_path / ".github" / "copilot-instructions.md"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"# Rules\r\n\r\nUse tabs.\r\n")
    before = p.read_bytes().count(b"\r\n")
    _inject_copilot_models_instructions(p, ["gpt-5.4"])
    _inject_copilot_models_instructions(p, ["gpt-5.5"])  # refresh path does the rewrite
    after = p.read_bytes()
    assert b"Use tabs." in after
    assert after.count(b"\r\n") >= before, "original CRLF endings were converted to LF"


def test_launch_survives_github_existing_as_a_file(tmp_path: Path) -> None:
    """`.github` as a FILE raised FileExistsError and killed the wrap pre-launch."""
    (tmp_path / ".github").write_text("not a directory", encoding="utf-8")
    target = tmp_path / ".github" / "copilot-instructions.md"
    with pytest.raises(OSError):
        _inject_copilot_models_instructions(target, ["gpt-5.4"])
    # The launch path must swallow exactly this, so assert it is an OSError
    # subclass (what the call site catches) rather than something broader.


# ---------------------------------------------------------------------------
# Ownership must be established by CONTENT, not position
# ---------------------------------------------------------------------------

_START = "<!-- headroom:available-models -->"
_END = "<!-- /headroom:available-models -->"
_KEEP = "DO NOT DELETE MY RULE"
_USER_PAIR = f"{_START}\n{_KEEP}\n{_END}\n"


def _real_block() -> str:
    from headroom.cli.wrap import _copilot_models_instructions_block

    return _copilot_models_instructions_block(["old-model"])


def _refresh_then_unwrap(path: Path) -> tuple[str, str]:
    """Two refreshes then a removal — the full lifecycle a user actually hits."""
    _inject_copilot_models_instructions(path, ["gpt-5.4"])
    _inject_copilot_models_instructions(path, ["gpt-5.5"])
    after = path.read_text(encoding="utf-8")
    _remove_copilot_models_instructions(path)
    return after, path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("label", "content", "keep"),
    [
        # Ownership cannot be inferred from anything IN the file. Each shape below
        # forges whatever the previous implementation trusted, and each destroyed
        # real user content before provenance-based ownership landed.
        #
        # Prose merely *describing* the feature: markers inline, heading quoted.
        (
            "prose documenting the feature",
            "# Team rules\n\nHeadroom keeps a `{S}` block here.\n"
            "The heading is `# Available AI models (Headroom)`.\n\n"
            "NEVER COMMIT SECRETS.\n\nIt closes with `{E}` -- do not edit.\n",
            "NEVER COMMIT SECRETS",
        ),
        # The canonical way anyone documents a marker-guarded block.
        (
            "markers inside a fenced code block",
            "# Docs\n\nHeadroom writes:\n\n```markdown\n{S}\n"
            "# Available AI models (Headroom)\n- `gpt-4`\n{KEEP}\n{E}\n```\n\ntail\n",
            None,
        ),
        # A signature test that is not line-anchored also matches "##".
        (
            "signature not line-anchored",
            "# Rules\n\n{S}\n## Available AI models (Headroom)\n{KEEP}\n{E}\n",
            None,
        ),
        (
            "markers with trailing whitespace",
            "# Rules\n\n{S}   \n## Available AI models (Headroom)\n{KEEP}\n{E}   \n",
            None,
        ),
        # The reviewer's original repro: a COMPLETE user pair above ours.
        ("complete user pair before", "# Rules\n\n{PAIR}\n{REAL}", None),
        ("complete user pair after", "# Rules\n\n{REAL}\n{PAIR}", None),
        ("user pairs on both sides", "{PAIR}\n{REAL}\n{PAIR}", None),
        ("stray START only", "# Rules\n\n{S}\n{KEEP}\n\n## section\n", None),
        ("stray END only", "# Rules\n\n{KEEP}\n{E}\n", None),
        ("reversed markers", "{E}\n{KEEP}\n{S}\n", None),
        ("user quotes our block verbatim", "# Rules\n{KEEP}\n\n{REAL}", None),
    ],
)
def test_user_content_survives_refresh_and_unwrap(
    tmp_path: Path, label: str, content: str, keep: str | None
) -> None:
    """The user's line must survive BOTH a refresh and an unwrap, in every shape."""
    sentinel = keep or _KEEP
    body = content.format(S=_START, E=_END, KEEP=_KEEP, PAIR=_USER_PAIR, REAL=_real_block())
    p = tmp_path / "copilot-instructions.md"
    p.write_text(body, encoding="utf-8")
    after, final = _refresh_then_unwrap(p)
    assert sentinel in after, f"{label}: user content destroyed by refresh"
    assert sentinel in final, f"{label}: user content destroyed by unwrap"


def test_lone_user_pair_with_no_generated_block_is_never_touched(tmp_path: Path) -> None:
    """Nothing of ours is present, so neither refresh nor unwrap may edit the pair."""
    p = tmp_path / "copilot-instructions.md"
    p.write_text(f"# Rules\n\n{_USER_PAIR}", encoding="utf-8")
    assert _remove_copilot_models_instructions(p) is False
    assert _KEEP in p.read_text(encoding="utf-8")
    after, final = _refresh_then_unwrap(p)
    assert _KEEP in after
    assert _KEEP in final


def test_duplicate_of_our_own_block_refuses_refresh_but_unwrap_still_cleans(
    tmp_path: Path,
) -> None:
    """Two byte-identical copies of OUR block: refresh is ambiguous, removal is not.

    Refusing to splice is right — there is no way to know which copy to update. But
    refusing to *remove* would wedge the file permanently and mean `unwrap copilot`
    cannot undo its own durable setup, so removal deletes every proven copy. That is
    safe precisely because each one is byte-identical to what we wrote.
    """
    p = tmp_path / "copilot-instructions.md"
    p.write_text("# Mine\nkeep me\n", encoding="utf-8")
    _inject_copilot_models_instructions(p, ["gpt-5.4"])
    text = p.read_text(encoding="utf-8")
    block = text[text.index(_START) :]
    p.write_text(text + "\n" + block, encoding="utf-8")
    assert p.read_text(encoding="utf-8").count(_START) == 2

    frozen = p.read_text(encoding="utf-8")
    assert _inject_copilot_models_instructions(p, ["gpt-5.5"]) is False
    assert p.read_text(encoding="utf-8") == frozen, "mutated while ambiguous"

    assert _remove_copilot_models_instructions(p) is True
    final = p.read_text(encoding="utf-8")
    assert _START not in final, "unwrap left a block behind (file would stay wedged)"
    assert "keep me" in final


def test_refresh_updates_in_place_without_duplicating(tmp_path: Path) -> None:
    """Normal operation, guarded against over-correcting into append-only."""
    p = tmp_path / "copilot-instructions.md"
    p.write_text("# Mine\nkeep me\n", encoding="utf-8")
    _inject_copilot_models_instructions(p, ["gpt-5.4"])
    assert "gpt-5.4" in p.read_text(encoding="utf-8")
    _inject_copilot_models_instructions(p, ["gpt-5.5"])
    text = p.read_text(encoding="utf-8")
    assert "gpt-5.5" in text
    assert "gpt-5.4" not in text, "stale list left behind"
    assert text.count(_START) == 1, "refresh appended instead of replacing"


@pytest.mark.parametrize("content", [b"# Mine\nkeep me\n", b"# Mine\r\nkeep me\r\n"])
def test_inject_then_remove_is_byte_identical(tmp_path: Path, content: bytes) -> None:
    """A source-controlled file must come back exactly as it was, endings included."""
    p = tmp_path / "copilot-instructions.md"
    p.write_bytes(content)
    _inject_copilot_models_instructions(p, ["gpt-5.4"])
    _remove_copilot_models_instructions(p)
    assert p.read_bytes() == content


def test_invalid_utf8_fails_closed(tmp_path: Path) -> None:
    """`errors="replace"` + whole-file rewrite would substitute U+FFFD everywhere."""
    p = tmp_path / "copilot-instructions.md"
    p.write_bytes(b"# Rules\n\xff\xfe raw bytes " + _KEEP.encode() + b"\n")
    before = p.read_bytes()
    assert _inject_copilot_models_instructions(p, ["gpt-5.4"]) is False
    assert p.read_bytes() == before
    assert _remove_copilot_models_instructions(p) is False
    assert p.read_bytes() == before


def test_utf16le_without_bom_fails_closed(tmp_path: Path) -> None:
    """UTF-16LE ASCII is valid UTF-8, so the decode guard alone admits it.

    The interleaved NUL bytes decode as U+0000, so a rewrite left the file
    undecodable in its own encoding. A BOM-prefixed UTF-16 file already failed
    closed (0xFF is invalid UTF-8); this covers the BOM-less case.
    """
    p = tmp_path / "copilot-instructions.md"
    p.write_bytes("# Mine\nkeep me\n".encode("utf-16-le"))
    before = p.read_bytes()
    assert _inject_copilot_models_instructions(p, ["gpt-5.4"]) is False
    assert p.read_bytes() == before
    assert _remove_copilot_models_instructions(p) is False
    assert p.read_bytes() == before
    # Still readable in its own encoding.
    assert p.read_bytes().decode("utf-16-le") == "# Mine\nkeep me\n"


# ---------------------------------------------------------------------------
# Crash-recoverable ownership record
# ---------------------------------------------------------------------------


class _PowerLoss(BaseException):
    """The process dying mid-write: no OSError handler gets to run."""


@pytest.fixture
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(workspace))
    return workspace


def _interrupt_nth_write(monkeypatch: pytest.MonkeyPatch, target: Path, nth: int) -> None:
    import headroom.cli.wrap as wrap_mod

    real = wrap_mod._write_text
    seen = {"n": 0}

    def _write_text(path: Path, content: str) -> None:
        if Path(path) == target:
            seen["n"] += 1
            if seen["n"] == nth:
                raise _PowerLoss(f"power lost before write #{nth} to {target.name}")
        real(path, content)

    monkeypatch.setattr(wrap_mod, "_write_text", _write_text)


def _blocks(path: Path) -> int:
    from headroom.cli.wrap import _COPILOT_MODELS_MARKER

    return path.read_text(encoding="utf-8").count(_COPILOT_MODELS_MARKER)


def test_refresh_interrupted_before_the_file_write_still_refreshes_in_place(
    instructions: Path, monkeypatch: pytest.MonkeyPatch, isolated_workspace: Path
) -> None:
    """Pending record written, file write never lands: the next launch must not duplicate."""
    _inject_copilot_models_instructions(instructions, ["a-1"])
    before = instructions.read_bytes()

    with monkeypatch.context() as m:
        _interrupt_nth_write(m, instructions, 1)
        with pytest.raises(_PowerLoss):
            _inject_copilot_models_instructions(instructions, ["a-1", "b-2"])
    assert instructions.read_bytes() == before

    assert _inject_copilot_models_instructions(instructions, ["a-1", "b-2", "c-3"]) is True
    assert _blocks(instructions) == 1, "an interrupted refresh made the next launch append"
    assert "c-3" in instructions.read_text(encoding="utf-8")
    assert _remove_copilot_models_instructions(instructions) is True
    assert _blocks(instructions) == 0


def test_refresh_interrupted_after_the_file_write_is_still_owned(
    instructions: Path, monkeypatch: pytest.MonkeyPatch, isolated_workspace: Path
) -> None:
    """New block landed, record never finalized: refresh and unwrap both still work."""
    from headroom.cli.wrap import _copilot_models_provenance_path

    _inject_copilot_models_instructions(instructions, ["a-1"])
    record = _copilot_models_provenance_path(instructions)
    with monkeypatch.context() as m:
        # Write #1 to the record is the pending record; #2 would finalize it.
        _interrupt_nth_write(m, record, 2)
        with pytest.raises(_PowerLoss):
            _inject_copilot_models_instructions(instructions, ["a-1", "b-2"])
    assert "b-2" in instructions.read_text(encoding="utf-8")

    snapshot, record_snapshot = instructions.read_bytes(), record.read_bytes()
    assert _inject_copilot_models_instructions(instructions, ["a-1", "b-2", "c-3"]) is True
    assert _blocks(instructions) == 1

    instructions.write_bytes(snapshot)
    record.write_bytes(record_snapshot)
    assert _remove_copilot_models_instructions(instructions) is True
    assert _blocks(instructions) == 0


def test_no_ownership_record_means_the_file_is_left_alone(
    instructions: Path, monkeypatch: pytest.MonkeyPatch, isolated_workspace: Path
) -> None:
    """A block Headroom could never prove it wrote would be duplicated on every launch."""
    import headroom.cli.wrap as wrap_mod

    blocker = isolated_workspace.parent / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(wrap_mod, "_copilot_models_provenance_path", lambda p: blocker / "r.json")
    before = instructions.read_bytes()

    assert _inject_copilot_models_instructions(instructions, ["a-1"]) is False
    assert instructions.read_bytes() == before


def test_a_v1_ownership_record_is_still_honoured(
    instructions: Path, isolated_workspace: Path
) -> None:
    import hashlib
    import json

    from headroom.cli.wrap import (
        _copilot_models_instructions_block,
        _copilot_models_provenance_path,
    )

    block = _copilot_models_instructions_block(["a-1"]).rstrip("\n")
    instructions.write_text(
        instructions.read_text(encoding="utf-8") + "\n" + block + "\n", encoding="utf-8"
    )
    record = _copilot_models_provenance_path(instructions)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {"path": str(instructions), "sha256": hashlib.sha256(block.encode()).hexdigest()}
        ),
        encoding="utf-8",
    )

    assert _inject_copilot_models_instructions(instructions, ["a-1", "b-2"]) is True
    assert _blocks(instructions) == 1
    assert json.loads(record.read_text(encoding="utf-8"))["version"] == 2
