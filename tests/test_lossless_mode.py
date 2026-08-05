"""Tests for no-CCR --lossless proxy mode (Stage A).

Covers:
  * flag plumbing (ProxyConfig field, CLI option parses),
  * reversibility of each format-native lossless compaction,
  * end-to-end ContentRouter(lossless=True) invariant: smaller output, NO
    ``<<ccr:`` / ``Retrieve `` marker, and full recoverability.
"""

from __future__ import annotations

from headroom.proxy.models import ProxyConfig
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)
from headroom.transforms.lossless_compaction import (
    collapse_runs,
    compact_lossless,
    diff_strip_index,
    expand_runs,
    is_run_collapsed,
    search_heading,
    search_unheading,
    strip_ansi,
)


# --------------------------------------------------------------------------
# Flag plumbing
# --------------------------------------------------------------------------
def test_proxyconfig_has_lossless_field() -> None:
    assert ProxyConfig().lossless is False
    assert ProxyConfig(lossless=True).lossless is True


def test_content_router_config_has_lossless_field() -> None:
    assert ContentRouterConfig().lossless is False
    cfg = ContentRouterConfig(lossless=True)
    assert cfg.lossless is True


def test_router_lossless_forces_marker_free_config() -> None:
    # Building the router normalizes the config so the no-CCR invariant holds
    # regardless of how ContentRouterConfig was constructed.
    router = ContentRouter(ContentRouterConfig(lossless=True))
    assert router.config.ccr_inject_marker is False
    assert router.config.smart_crusher_lossless_only is True


def test_cli_proxy_option_parses_lossless() -> None:
    from click.testing import CliRunner

    from headroom.cli.proxy import proxy

    runner = CliRunner()
    result = runner.invoke(proxy, ["--help"])
    assert result.exit_code == 0
    assert "--lossless" in result.output


# --------------------------------------------------------------------------
# Reversibility: collapse_runs / expand_runs
# --------------------------------------------------------------------------
def test_collapse_expand_runs_byte_roundtrip() -> None:
    log = (
        "starting worker\n"
        "connection refused\n"
        "connection refused\n"
        "connection refused\n"
        "connection refused\n"
        "connection refused\n"
        "retrying\n"
        "retrying\n"
        "done\n"
    )
    collapsed = collapse_runs(log)
    assert is_run_collapsed(collapsed)
    assert len(collapsed) < len(log)
    assert expand_runs(collapsed) == log


def test_collapse_runs_no_trailing_newline_roundtrip() -> None:
    log = "a\na\na\nb"
    assert expand_runs(collapse_runs(log)) == log


def test_collapse_runs_singletons_untouched() -> None:
    log = "one\ntwo\nthree\n"
    assert collapse_runs(log) == log
    assert not is_run_collapsed(log)


def test_collapse_runs_empty() -> None:
    assert collapse_runs("") == ""
    assert expand_runs("") == ""


# --------------------------------------------------------------------------
# Reversibility: search_heading / search_unheading
# --------------------------------------------------------------------------
def test_search_heading_unheading_roundtrip() -> None:
    grep = (
        "src/app.py:10:def main():\n"
        "src/app.py:11:    run()\n"
        "src/app.py:42:    return 0\n"
        "src/util.py:3:import os\n"
        "src/util.py:9:import sys\n"
    )
    headed = search_heading(grep)
    # heading form: each path appears once as its own header line
    assert headed.count("src/app.py") == 1
    assert headed.count("src/util.py") == 1
    assert search_unheading(headed) == grep


def test_search_heading_smaller_for_repeated_paths() -> None:
    grep = "\n".join(f"a/very/long/path/module.py:{i}:line{i}" for i in range(1, 30)) + "\n"
    headed = search_heading(grep)
    assert len(headed) < len(grep)
    assert search_unheading(headed) == grep


def test_search_heading_leaves_non_matching_lines() -> None:
    text = "just some prose\nnot a grep row at all\n"
    assert search_heading(text) == text
    assert search_unheading(text) == text


def test_search_heading_mixed_content_roundtrip() -> None:
    grep = "banner line\nsrc/a.py:1:x\nsrc/a.py:2:y\nmiddle prose\nsrc/b.py:5:z\n"
    headed = search_heading(grep)
    assert search_unheading(headed) == grep


# --------------------------------------------------------------------------
# strip_ansi
# --------------------------------------------------------------------------
def test_strip_ansi_removes_only_escapes() -> None:
    colored = "\x1b[31mERROR\x1b[0m: boom \x1b[1mbold\x1b[0m end"
    assert strip_ansi(colored) == "ERROR: boom bold end"
    plain = "no escapes here : 1:2:3"
    assert strip_ansi(plain) == plain


# --------------------------------------------------------------------------
# diff_strip_index
# --------------------------------------------------------------------------
def test_diff_strip_index_keeps_plus_minus() -> None:
    diff = (
        "diff --git a/f.py b/f.py\n"
        "index 0123abc..def4567 100644\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,3 +1,3 @@\n"
        " context\n"
        "-old line\n"
        "+new line\n"
    )
    stripped = diff_strip_index(diff)
    assert "index 0123abc..def4567" not in stripped
    assert "-old line" in stripped
    assert "+new line" in stripped
    assert "@@ -1,3 +1,3 @@" in stripped
    assert "--- a/f.py" in stripped
    assert len(stripped) < len(diff)


# --------------------------------------------------------------------------
# compact_lossless dispatch + safety gates
# --------------------------------------------------------------------------
def test_compact_lossless_log_roundtrips_modulo_ansi() -> None:
    log = "\x1b[31mfail\x1b[0m\n" + "fail\n" * 5
    out = compact_lossless(log, "log")
    # recoverable modulo ANSI: expand back == de-ANSI'd original
    assert expand_runs(out) == strip_ansi(log)
    assert len(out) < len(log)


def test_compact_lossless_returns_original_when_not_smaller() -> None:
    # No repeats, no ANSI -> nothing to gain; returns original unchanged.
    log = "line one\nline two\nline three\n"
    assert compact_lossless(log, "log") == log


def test_compact_lossless_search() -> None:
    grep = "\n".join(f"pkg/mod.py:{i}:code{i}" for i in range(1, 20)) + "\n"
    out = compact_lossless(grep, "search")
    assert len(out) < len(grep)
    assert search_unheading(out) == grep


def test_compact_lossless_unknown_kind_passthrough() -> None:
    assert compact_lossless("whatever", "mystery") == "whatever"


def test_compact_lossless_never_raises_on_empty() -> None:
    for kind in ("log", "search", "diff", "text"):
        assert compact_lossless("", kind) == ""


# --------------------------------------------------------------------------
# End-to-end: ContentRouter(lossless=True) invariant
# --------------------------------------------------------------------------
def _assert_no_marker(text: str) -> None:
    assert "<<ccr:" not in text
    assert "Retrieve " not in text


def test_router_lossless_log_strategy_no_marker_and_recoverable() -> None:
    # Drive the LOG strategy directly: detection is content-dependent, but the
    # router's lossless disposition for an explicit LOG strategy must apply
    # format-native lossless compaction with no marker and full recovery.
    router = ContentRouter(ContentRouterConfig(lossless=True))
    log = (
        "[info] boot sequence start\n"
        + "connection refused by upstream service alpha\n" * 40
        + "[info] boot sequence complete\n"
    )
    out, _tokens, chain = router._apply_strategy_to_content(
        log, CompressionStrategy.LOG, context=""
    )
    _assert_no_marker(out)
    assert len(out) < len(log)
    assert chain == ["lossless_log"]
    # every original line recoverable (no ANSI here, so exact)
    assert expand_runs(out) == log


def test_router_lossless_diff_strategy_no_marker() -> None:
    router = ContentRouter(ContentRouterConfig(lossless=True))
    diff = (
        "diff --git a/f.py b/f.py\n"
        "index 0123abc..def4567 100644\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
    )
    out, _tokens, chain = router._apply_strategy_to_content(
        diff, CompressionStrategy.DIFF, context=""
    )
    _assert_no_marker(out)
    assert chain == ["lossless_diff"]
    assert "index 0123abc..def4567" not in out
    assert "-old" in out and "+new" in out


def test_router_lossless_search_no_marker_and_recoverable() -> None:
    router = ContentRouter(ContentRouterConfig(lossless=True))
    grep = (
        "\n".join(
            f"headroom/transforms/content_router.py:{i}:    identifier_{i} = compute()"
            for i in range(1, 60)
        )
        + "\n"
    )
    result = router.compress(grep, context="")
    out = result.compressed
    _assert_no_marker(out)
    # identifiers still present / recoverable
    if result.strategy_used == CompressionStrategy.SEARCH:
        assert search_unheading(out) == grep
    # at minimum, no data lost and no marker
    for i in (1, 30, 59):
        assert f"identifier_{i}" in out


def test_router_lossless_never_emits_marker_various_inputs() -> None:
    router = ContentRouter(ContentRouterConfig(lossless=True))
    samples = [
        "err\n" * 100,
        "\n".join(f"a/b/c.py:{i}:x{i}" for i in range(50)),
        "plain prose line\n" * 30,
    ]
    for s in samples:
        _assert_no_marker(router.compress(s, context="").compressed)


def test_router_apply_accepts_lossless_search_token_measured() -> None:
    """Regression: the acceptance gate in router.apply() measured WORD count, so
    a lossless search fold — which cuts TOKENS by collapsing a repeated path
    prefix while word count stays flat or *rises* (the heading adds a word) — was
    wrongly discarded as ratio_too_high. The gate now measures lossless results
    by real token count, so the free, recoverable win is applied. (compress()/
    _apply_strategy_to_content bypass this gate, which is why unit tests above
    never caught it — the bug only appears through the full apply() path.)
    """
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer
    from headroom.transforms.lossless_compaction import search_heading

    tok = Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")
    router = ContentRouter(ContentRouterConfig(lossless=True))
    grep = "".join(
        f"headroom/transforms/content_router.py:{i}:    identifier_{i} = compute(value)\n"
        for i in range(1, 60)
    )
    # The fold does NOT reduce word count (the heading even adds one) — this is
    # exactly what made the old word-count gate reject it.
    assert len(search_heading(grep).split()) >= len(grep.split())

    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "find_refs", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": grep},
    ]
    out = router.apply(messages, tok).messages[1]["content"]
    assert tok.count_text(out) < tok.count_text(grep)  # accepted: fewer TOKENS
    assert search_unheading(out) == grep  # byte-exact recovery
    _assert_no_marker(out)


# --------------------------------------------------------------------------
# Token-delta measurement (informational)
# --------------------------------------------------------------------------
def test_measure_token_deltas(capsys) -> None:  # type: ignore[no-untyped-def]
    log = (
        "[warn] disk usage high\n"
        + "\x1b[33mretrying connection to db-primary\x1b[0m\n" * 50
        + "[info] recovered\n"
    )
    grep = (
        "\n".join(
            f"headroom/proxy/server.py:{i}:    router_config.value_{i} = {i}" for i in range(1, 80)
        )
        + "\n"
    )
    log_out = compact_lossless(log, "log")
    grep_out = compact_lossless(grep, "search")
    # chars are what the byte-size gate optimizes and what subword tokenizers
    # track closely; whitespace word-count is reported for direction only.
    log_c0, log_c1 = len(log), len(log_out)
    grep_c0, grep_c1 = len(grep), len(grep_out)
    log_t0, log_t1 = len(log.split()), len(log_out.split())
    grep_t0, grep_t1 = len(grep.split()), len(grep_out.split())
    print(
        f"\n[lossless deltas] "
        f"log: {log_c0}->{log_c1} chars "
        f"({100 * (log_c0 - log_c1) / log_c0:.1f}% saved), {log_t0}->{log_t1} words; "
        f"grep: {grep_c0}->{grep_c1} chars "
        f"({100 * (grep_c0 - grep_c1) / grep_c0:.1f}% saved), {grep_t0}->{grep_t1} words"
    )
    assert log_c1 < log_c0
    assert grep_c1 < grep_c0


def test_lossless_mode_builds_kompress_marker_free(monkeypatch) -> None:
    """In lossless (no-CCR) mode Kompress must be built with enable_ccr=False so
    it never appends a `Retrieve more: hash=` marker or writes the CCR store —
    the agent has no MCP tool to redeem it. Kompress still runs (lossy); only
    the marker/store is suppressed. Verified WITHOUT loading the model.
    """
    import headroom.transforms.kompress_compressor as kc
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    captured: dict[str, object] = {}

    class _FakeKompress:
        def __init__(self, config=None) -> None:
            captured["enable_ccr"] = getattr(config, "enable_ccr", None)

        def is_ready(self) -> bool:
            return False

        def ensure_background_load(self) -> None:
            pass

    monkeypatch.setattr(kc, "is_kompress_available", lambda: True)
    monkeypatch.setattr(kc, "KompressCompressor", _FakeKompress)

    ContentRouter(ContentRouterConfig(lossless=True))._get_kompress()
    assert captured["enable_ccr"] is False  # marker suppressed in lossless mode

    captured.clear()
    ContentRouter(ContentRouterConfig(lossless=False))._get_kompress()
    assert captured["enable_ccr"] is True  # normal mode: unchanged (marker on)
