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


def _ruby_logger_log() -> str:
    """Ruby's default ``Logger::Formatter`` at every severity (UNKNOWN prints as ANY)."""
    return "\n".join(
        f"{sev[0]}, [2026-09-02T14:{m:02d}:10.{m:06d} #4321] {sev:>5} -- : request {m} handled"
        for sev in ("DEBUG", "INFO", "WARN", "ERROR", "FATAL", "ANY")
        for m in range(10)
    )


_LONGEST_HOST = ".".join(["h" * 63] * 3 + ["h" * 61])  # 253 chars, the DNS maximum


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
        "\n".join(f"2026/09/02 14:{m:02d}:10 ERROR upstream timeout req={m}" for m in range(60)),
        "\n".join(
            f"2026/09/02 14:{m:02d}:10 [error] 31#31: *{m} connect() failed (111: refused)"
            for m in range(60)
        ),
        "\n".join(
            f"E0902 14:{m:02d}:10.000000       1 reflector.go:{m}] failed to list *v1.Pod"
            for m in range(60)
        ),
        "\n".join(
            f'10.0.0.7 - - [02/Sep/2026:14:{m:02d}:10 +0000] "GET /items/{m} HTTP/1.1" 500 512'
            for m in range(60)
        ),
        "\n".join(
            f'{_LONGEST_HOST} - - [02/Sep/2026:14:{m:02d}:10 +0000] "GET /" 200 5'
            for m in range(60)
        ),
        _ruby_logger_log(),
    ],
    ids=[
        "iso-datetime",
        "syslog",
        "bracketed-time",
        "iso-T",
        "go-log",
        "nginx-error",
        "klog",
        "access-log",
        "access-log-longest-host",
        "ruby-logger",
    ],
)
def test_timestamped_logs_are_never_folded(sample: str) -> None:
    assert compact_lossless(sample, "search") == sample


@pytest.mark.parametrize(
    ("path", "row"),
    [
        ("2026/09/02/api.log", "2026/09/02 14:30:{s:02d} ERROR upstream timeout"),
        ("logs/nginx/error.log", "2026/09/02 14:30:{s:02d} [error] 31#31: connect() failed"),
        ("logs/controller.log", "E0902 14:30:{s:02d}.000000       1 reflector.go:100] list failed"),
        ("/var/log/nginx/access.log", '10.0.0.7 - - [02/Sep/2026:14:30:{s:02d} +0000] "GET /" 500'),
        ("log/production.log", "I, [2026-09-02T14:30:{s:02d}.000000 #1]  INFO -- : 200 OK"),
    ],
    ids=["go-log", "nginx-error", "klog", "access-log", "ruby-logger"],
)
def test_grep_rows_over_those_logs_still_fold(path: str, row: str) -> None:
    """``grep -n`` output starts with the file path, so it is not a log row."""
    grep = "\n".join(f"{path}:{i}:{row.format(s=i % 60)}" for i in range(1, 41))
    out = compact_lossless(grep, "search")
    assert len(out) < len(grep)
    assert out.split("\n")[0] == path


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
    grep = "\n".join(f"src/a.py:{i}:    x = {i}" for i in range(40))
    prior = os.environ.pop("HEADROOM_LOSSLESS_COMPACTION", None)
    try:
        assert compact_lossless(grep, "search") != grep
        os.environ["HEADROOM_LOSSLESS_COMPACTION"] = "off"
        assert compact_lossless(grep, "search") == grep
    finally:
        os.environ.pop("HEADROOM_LOSSLESS_COMPACTION", None)
        if prior is not None:
            os.environ["HEADROOM_LOSSLESS_COMPACTION"] = prior
