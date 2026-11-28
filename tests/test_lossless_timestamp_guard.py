"""Timestamped log lines must never be folded as grep ``path:line:content`` rows.

``2026-09-02 14:30:00 [FATAL] ...`` splits under _GREP_ROW_RE as
path=``2026-09-02 14`` / line=``30`` / content=``00 [FATAL] ...``. Folding it
hoists the date+hour into a heading and strips it from every row, so the model
receives ``30:00 [FATAL] ...`` and has to rebuild the clock itself. The fold is
byte-reversible, so compact_lossless's inverse-check cannot catch it.
"""

import os

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
    "bracket": "worker[",
}


def _prefixed_log(prefix: str, hours: int = 3) -> list[str]:
    return [
        f"{prefix}{h}:{m:02d}:10 record {h}-{m}" for h in range(9, 9 + hours) for m in range(60)
    ]


@pytest.mark.parametrize("prefix", _CLOCK_PREFIXES.values(), ids=_CLOCK_PREFIXES)
def test_clock_behind_a_prefix_is_not_split(prefix: str) -> None:
    lines = _prefixed_log(prefix)
    text = "\n".join(lines)
    out = compact_lossless(text, "search")
    # One heading for all three hours, and every row starts with its whole clock.
    assert out.split("\n") == [prefix] + [line[len(prefix) :] for line in lines]
    assert search_unheading(out, clocks=True) == text


def test_a_clock_row_that_shares_its_prefix_with_neither_neighbour_stays_whole() -> None:
    a, b = _prefixed_log("worker[", 1), _prefixed_log("db[", 1)
    text = "\n".join(line for pair in zip(a, b) for line in pair)
    assert search_heading(text, clocks=True) == text
    # Cut at the first colon instead, each line gets a heading of its own: no smaller.
    assert len(search_heading(text)) == len(text)
    assert compact_lossless(text, "search") == text
    text = "\n".join([b[0], *a[:3], b[1], *a[3:6]])
    out = search_heading(text, clocks=True)
    group = [["worker[", *(line[7:] for line in lines)] for lines in (a[:3], a[3:6])]
    assert out.split("\n") == [b[0], *group[0], b[1], *group[1]]
    assert search_unheading(out, clocks=True) == text


def test_clock_headings_and_grep_headings_mix() -> None:
    grep = [f"src/app.py:{n}:30:10 is the row's own text" for n in (14, 15, 16)]
    text = "\n".join(
        [*_prefixed_log("ts=", 1)[:3], *grep, *_prefixed_log("worker[", 1)[:3], "plain"]
    )
    out = search_heading(text, clocks=True)
    assert out.split("\n")[4:8] == ["src/app.py", *(line.split(":", 1)[1] for line in grep)]
    assert search_unheading(out, clocks=True) == text


def test_search_heading_without_clocks_is_unchanged() -> None:
    text = "\n".join(_prefixed_log("worker[", 1)[:3])
    out = search_heading(text)
    assert out == "worker[9\n00:10 record 9-0\n01:10 record 9-1\n02:10 record 9-2"
    assert search_unheading(out) == text
    # Only clocks=True reads a row under a clock heading as a clock.
    assert search_unheading("notes=\n14:30:10 x") == "notes=:14:30:10 x"
    assert search_unheading("notes=\n14:30:10 x", clocks=True) == "notes=14:30:10 x"
    assert search_unheading("notes=\n14:x", clocks=True) == "notes=:14:x"
    assert search_unheading("notes\n14:30:10 x", clocks=True) == "notes:14:30:10 x"


_CLOCK_LIKE_FILES = [
    "results,14",
    "my notes 14",
    "ts=14",
    "x[14",
    'say "14',
    "run 2026-11-24 14",
    "app-2026-09-02T14",
    "a, 2026 9",
]


@pytest.mark.parametrize("path", _CLOCK_LIKE_FILES)
@pytest.mark.parametrize("block", [1, 2, 5], ids=["alternating", "pairs", "blocks"])
def test_mixed_rows_of_a_file_named_like_a_clock_fold_by_file(path: str, block: int) -> None:
    """One row that is no clock shows the path is a file, so it folds as it always did."""
    rows = ["10 items" if (i // block) % 2 == 0 else "normal text" for i in range(50)]
    grep = "\n".join(f"{path}:{10 + i}:{row} {i}" for i, row in enumerate(rows))
    out = compact_lossless(grep, "search")
    assert out == search_heading(grep)
    assert out.split("\n")[0] == path
    assert len(out) < len(grep)
    assert search_unheading(out) == grep


@pytest.mark.parametrize(
    ("path", "content", "lines"),
    [
        ("logs/2026/11/24", "30 items", range(15, 55)),
        ("chunk12", "30 files", range(15, 55)),
        ("part-7", "45 rows", range(15, 55)),
        ("my notes 14", "30 items", range(15, 55)),  # a bare space opens no clock
        ("results,14", "30 items", range(15, 55)),
        ("at 2026-11-24 14", "75 items", range(15, 55)),  # 75 is not a second
        ("at 2026-11-24 14", "105 items", range(15, 55)),
        ("at 2026-11-24 14", "10 items", range(1, 10)),  # a one-digit minute is not a clock
        ("at 2026-11-24 3", "30 items", range(60, 100)),  # 60 and up is not a minute
        ("at 2026-11-24 141", "30 items", range(15, 55)),  # three digits are not an hour
        ("at 26-11-24 14", "30 items", range(15, 55)),  # not an ISO date
        ("v1.2026 14", "30 items", range(15, 55)),  # a year follows ", "
        ("REPORT14", "30 items", range(15, 55)),  # a T opens a clock only after a date
    ],
)
def test_grep_rows_that_only_resemble_a_clock_fold_by_path(path, content, lines) -> None:
    grep = "\n".join(f"{path}:{lines[i % len(lines)]}:{content}" for i in range(40))
    out = search_heading(grep, clocks=True)
    assert out == search_heading(grep)
    assert out.split("\n")[0] == path
    assert compact_lossless(grep, "search") == out


@pytest.mark.parametrize("prefix", ["web-1  | ", "svc (", "a,b,"])
def test_other_prefixes_fold_as_before(prefix: str) -> None:
    text = "\n".join(_prefixed_log(prefix))
    assert compact_lossless(text, "search") == search_heading(text)


def test_when_only_a_split_clock_folds_it_is_still_taken() -> None:
    # Joined back without its ":", the last row would read `notes=14:30:10`, so the
    # whole-clock fold is not exact here. Not folding at all would be worse: the rows
    # would go on to the lossy compressors.
    text = "\n".join([*_prefixed_log("worker[", 1), "notes=:14:30:10 x"])
    out = compact_lossless(text, "search")
    assert out == search_heading(text)
    assert len(out) < len(text)
