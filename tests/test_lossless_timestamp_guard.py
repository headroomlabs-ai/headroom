"""Timestamped log lines must never be folded as grep ``path:line:content`` rows.

``2026-09-02 14:30:00 [FATAL] ...`` splits under _GREP_ROW_RE as
path=``2026-09-02 14`` / line=``30`` / content=``00 [FATAL] ...``. Folding it
hoists the date+hour into a heading and strips it from every row, so the model
receives ``30:00 [FATAL] ...`` and has to rebuild the clock itself. The fold is
byte-reversible, so compact_lossless's inverse-check cannot catch it.
"""

import os
import re

import pytest

from headroom.transforms.lossless_compaction import (
    compact_lossless,
    search_heading,
    search_unheading,
)


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


# --- a clock behind a prefix: folded, but cut before the hour ---
_CLOCK_PREFIXES = {
    "git-blame": "c75ebdee6 (Dev One  2026-07-02 ",
    "java-util-logging": "Nov 24, 2026 ",
    "iso-t": "build-7 2026-09-02T",
    "logfmt-iso": 'time="2026-09-02T',
    "logfmt-clock": "ts=",
    "quoted-clock": 'time="',
    "single-quoted-clock": "t='",
    "bracket": "worker[",
    "paren": "svc (",
    "comma": "a,b,",
    "pipe": "web|",
    "tab": "pod\t",
}


def _prefixed_log(prefix: str, hours: int = 3) -> list[str]:
    return [
        f"{prefix}{h}:{m:02d}:10 record {h}-{m}" for h in range(9, 9 + hours) for m in range(60)
    ]


@pytest.mark.parametrize("prefix", _CLOCK_PREFIXES.values(), ids=_CLOCK_PREFIXES)
def test_clock_behind_a_prefix_is_not_split(prefix: str) -> None:
    lines = _prefixed_log(prefix)
    text = "\n".join(lines)
    out = compact_lossless(text, "search").split("\n")
    # One heading for all three hours, and every row starts with its whole clock.
    assert out == [prefix] + [line[len(prefix) :] for line in lines]
    assert search_unheading("\n".join(out)) == text


def test_a_line_that_shares_its_prefix_with_neither_neighbour_stays_whole() -> None:
    a, b = _prefixed_log("worker[", 1), _prefixed_log("db[", 1)
    text = "\n".join(line for pair in zip(a, b) for line in pair)
    assert search_heading(text) == text
    text = "\n".join([b[0], *a[:3], b[1], *a[3:6]])
    out = search_heading(text)
    group = [["worker[", *(line[7:] for line in lines)] for lines in (a[:3], a[3:6])]
    assert out.split("\n") == [b[0], *group[0], b[1], *group[1]]
    assert search_unheading(out) == text


def test_clock_headings_and_grep_headings_mix() -> None:
    grep = [f"src/app.py:{n}:30:10 is the row's own text" for n in (14, 15, 16)]
    text = "\n".join(
        [*_prefixed_log("ts=", 1)[:3], *grep, *_prefixed_log("worker[", 1)[:3], "plain"]
    )
    out = search_heading(text)
    assert out.split("\n")[4:8] == ["src/app.py", *(line.split(":", 1)[1] for line in grep)]
    assert search_unheading(out) == text


@pytest.mark.parametrize("path", ["notes[", "a b T1T", "x="])
def test_grep_row_that_would_decode_as_a_clock_stays_whole(path: str) -> None:
    # Joined back without its ":", this row would read `notes[14:30:10`.
    text = "\n".join([f"{path}:14:30:10 text"] * 3 + _prefixed_log("worker[", 1))
    out = compact_lossless(text, "search")
    assert len(out) < len(text)
    assert search_unheading(out) == text


@pytest.mark.parametrize("path", ["notes[", "a b T1T", "x="])
def test_grep_rows_under_a_path_that_ends_like_a_clock_heading_still_fold(path: str) -> None:
    grep = "\n".join(f"{path}:{n}:plain text" for n in range(1, 41))
    out = compact_lossless(grep, "search")
    assert out.split("\n")[0] == path
    assert search_unheading(out) == grep


@pytest.mark.parametrize(
    ("path", "content", "lines"),
    [
        ("logs/2026/11/24", "30 items", range(15, 55)),  # "24" follows a "/"
        ("chunk12", "30 files", range(15, 55)),  # "12" is part of a name
        ("part-7", "45 rows", range(15, 55)),
        ("v2.10", "15 entries", range(15, 55)),
        ("deploy 14", "75 items", range(15, 55)),  # 75 is not a second
        ("deploy 14", "105 items", range(15, 55)),
        ("deploy 14", "10 items", range(1, 10)),  # a one-digit minute is not a clock
        ("tasks 3", "30 items", range(60, 100)),  # 60 and up is not a minute
        ("deploy 141", "30 items", range(15, 55)),  # three digits are not an hour
    ],
)
def test_grep_rows_that_only_resemble_a_clock_fold_by_path(path, content, lines) -> None:
    grep = "\n".join(f"{path}:{lines[i % len(lines)]}:{content}" for i in range(40))
    out = search_heading(grep)
    assert out.split("\n")[0] == path
    assert search_unheading(out) == grep


def test_prefixed_logs_never_lose_the_fold_to_a_split_clock() -> None:
    """Two prefixes, changing every few lines: still smaller, still exact, clocks whole."""
    lines = [
        f"{'worker[' if (i // 3) % 2 else 'db-1 ['}{10 + i // 60}:{i % 60:02d}:10] r{i}"
        for i in range(180)
    ]
    text = "\n".join(lines)
    out = compact_lossless(text, "search")
    assert len(out) < len(text)
    assert search_unheading(out) == text
    rows = [row for row in out.split("\n") if re.match(r"\d", row)]
    assert len(rows) == len(lines)
    assert all(re.match(r"\d{1,2}:\d{2}:\d{2}] r\d+$", row) for row in rows)
