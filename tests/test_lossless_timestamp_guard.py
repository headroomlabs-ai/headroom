"""Timestamped log lines must never be folded as grep ``path:line:content`` rows.

``2026-09-02 14:30:00 [FATAL] ...`` splits under _GREP_ROW_RE as
path=``2026-09-02 14`` / line=``30`` / content=``00 [FATAL] ...``. Folding it
hoists the date+hour into a heading and strips it from every row, so the model
receives ``30:00 [FATAL] ...`` and has to rebuild the clock itself. The fold is
byte-reversible, so compact_lossless's inverse-check cannot catch it.
"""

import os

import pytest

from headroom.transforms.lossless_compaction import compact_lossless


def _multi_hour_log() -> str:
    lines = [
        f"2026-09-02 {h:02d}:{m:02d}:10 [INFO] worker heartbeat ok id={m}"
        for h in (14, 15, 16)
        for m in range(60)
    ]
    lines.insert(90, "2026-09-02 14:30:00 [FATAL] node evicted disk full")
    return "\n".join(lines)


@pytest.mark.parametrize(
    "sample",
    [
        _multi_hour_log(),
        "\n".join(
            f"Aug 16 11:{m:02d}:22 web-01 systemd[3397]: Starting example-worker.service {m}"
            for m in range(60)
        ),
        "\n".join(f"[15:{m:02d}:53] tool: Bash(cmd {m})" for m in range(60)),
        "\n".join(f"2026-09-02T14:{m:02d}:10 [WARN] latency high {m}" for m in range(60)),
    ],
    ids=["iso-datetime", "syslog", "bracketed-time", "iso-T"],
)
def test_timestamped_logs_are_never_folded(sample: str) -> None:
    assert compact_lossless(sample, "search") == sample


def test_fatal_line_keeps_its_full_timestamp() -> None:
    out = compact_lossless(_multi_hour_log(), "search")
    fatal = [line for line in out.split("\n") if "FATAL" in line]
    assert fatal == ["2026-09-02 14:30:00 [FATAL] node evicted disk full"]


def test_genuine_grep_output_still_folds() -> None:
    """The guard must not cost us the real feature."""
    grep = "\n".join(
        f"src/lumina_tray/panel.py:{i}:    def handler_{i}(self): pass" for i in range(40)
    )
    out = compact_lossless(grep, "search")
    assert out != grep
    assert len(out) < len(grep)
    assert out.split("\n")[0] == "src/lumina_tray/panel.py"


def test_env_kill_switch_disables_all_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    grep = "\n".join(
        f"src/lumina_tray/panel.py:{i}:    def handler_{i}(self): pass" for i in range(40)
    )
    monkeypatch.setenv("HEADROOM_LOSSLESS_COMPACTION", "0")
    assert compact_lossless(grep, "search") == grep
    monkeypatch.setenv("HEADROOM_LOSSLESS_COMPACTION", "1")
    assert compact_lossless(grep, "search") != grep


def test_kill_switch_is_read_live_not_cached_at_import() -> None:
    """The proxy hot-syncs runtime env; a cached flag would ignore it."""
    grep = "\n".join(
        f"src/a.py:{i}:    x = {i}" for i in range(40)
    )
    prior = os.environ.pop("HEADROOM_LOSSLESS_COMPACTION", None)
    try:
        assert compact_lossless(grep, "search") != grep
        os.environ["HEADROOM_LOSSLESS_COMPACTION"] = "off"
        assert compact_lossless(grep, "search") == grep
    finally:
        os.environ.pop("HEADROOM_LOSSLESS_COMPACTION", None)
        if prior is not None:
            os.environ["HEADROOM_LOSSLESS_COMPACTION"] = prior
