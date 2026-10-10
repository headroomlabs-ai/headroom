"""Tests for the payload-safe log helpers in ``headroom.log_safety``."""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import py_compile
import sys
import types
import zipfile
from pathlib import Path

import pytest

from headroom import log_safety
from headroom.log_safety import WarnOnce, describe_exception, redact_url, safe_id

CANARY = "sk-live-CANARY tool output: patient record 42"


def _raise_chain() -> BaseException:
    try:
        try:
            raise KeyError(CANARY)
        except KeyError as inner:
            raise ValueError(f"could not parse {CANARY}") from inner
    except ValueError as exc:
        return exc


def test_exception_description_names_types_and_places_but_not_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    text = describe_exception(_raise_chain())

    assert "CANARY" not in text
    assert text.startswith("ValueError at ")
    assert "test_log_safety.py:" in text
    assert "in _raise_chain" in text
    assert "; caused by KeyError at " in text


def test_oserror_keeps_errno_but_not_its_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    text = describe_exception(PermissionError(13, "Permission denied", f"/tmp/{CANARY}"))

    assert text == "PermissionError [Errno 13] Permission denied"


def test_oserror_free_text_strerror_is_replaced_by_the_system_text() -> None:
    text = describe_exception(OSError(2, f"token={CANARY}"))

    assert "CANARY" not in text
    assert text == f"FileNotFoundError [Errno 2] {os.strerror(2)}"


def test_content_opt_in_gives_the_full_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")

    text = describe_exception(_raise_chain())

    assert "Traceback" in text
    assert CANARY in text


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://user:hunter2@proxy.example:8443/v1?key=abc#frag",
            "https://proxy.example:8443/<path>?<redacted>",
        ),
        ("https://upstream.example/botSECRET/v1/messages", "https://upstream.example/<path>"),
        ("https://upstream.example/bot%53ECRET/v1", "https://upstream.example/<path>"),
        ("http://127.0.0.1:8787", "http://127.0.0.1:8787"),
        ("http://[::1]:8787/", "http://[::1]:8787/"),
        ("http://[::1", "<unparseable url>"),
    ],
)
def test_redact_url_keeps_only_scheme_host_and_port(
    url: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    assert redact_url(url) == expected


def test_content_opt_in_keeps_the_path_but_never_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")

    assert redact_url("https://u:pw@upstream.example/v1/messages?key=abc") == (
        "https://upstream.example/v1/messages?<redacted>"
    )


def test_safe_id_escapes_newlines_and_bounds_length() -> None:
    assert safe_id("m1\nWARNING forged") == "'m1\\nWARNING forged'"
    long = safe_id("x" * 10_000)
    assert len(long) < 120
    assert long.endswith("chars)")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_warn_once_stays_bounded_across_repeated_passes(caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("headroom.test_log_safety.bounded")
    guard = WarnOnce(limit=64, what="unwritable ledgers")

    with caplog.at_level(logging.INFO, logger=log.name):
        for _ in range(3):
            for path in range(65):
                if guard.first(path, log):
                    log.warning("ledger %d failed", path)

    messages = _warnings(caplog)
    assert len(messages) == 65  # 64 keys plus one overflow notice, not 65 per pass
    assert messages[-1] == (
        "More than 64 warnings about unwritable ledgers this window; further ones are not logged "
        "at WARNING "
        "for the next 60 minutes"
    )


def test_warn_once_keeps_the_key_while_warnings_are_disabled() -> None:
    log = logging.getLogger("headroom.test_log_safety.level")
    guard = WarnOnce(limit=4, what="failures")
    log.setLevel(logging.ERROR)
    try:
        assert guard.first("k", log) is False
        log.setLevel(logging.INFO)
        assert guard.first("k", log) is True
        assert guard.first("k", log) is False
    finally:
        log.setLevel(logging.NOTSET)


def test_forget_rearms_a_key() -> None:
    log = logging.getLogger("headroom.test_log_safety.forget")
    guard = WarnOnce(limit=4, what="failures")

    assert guard.first("path", log) is True
    guard.forget("path")
    assert guard.first("path", log) is True


def test_frame_paths_are_package_relative_for_headroom_and_bare_otherwise() -> None:
    with pytest.raises(ValueError) as headroom_error:
        WarnOnce(limit=0, what="failures")
    with pytest.raises(json.JSONDecodeError) as stdlib_error:
        json.loads("{")

    assert "at headroom/log_safety.py:" in describe_exception(headroom_error.value)
    assert " in __init__" in describe_exception(headroom_error.value)
    assert "decoder.py:" in describe_exception(stdlib_error.value)


def test_raise_from_none_hides_the_suppressed_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    try:
        try:
            raise KeyError(CANARY)
        except KeyError:
            raise ValueError("bad") from None
    except ValueError as exc:
        text = describe_exception(exc)

    assert "KeyError" not in text


def test_a_cause_cycle_is_described_once() -> None:
    first = ValueError(CANARY)
    second = KeyError(CANARY)
    first.__cause__ = second
    second.__cause__ = first

    assert describe_exception(first) == "ValueError; caused by KeyError"


def test_forgetting_after_overflow_does_not_reopen_the_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("headroom.test_log_safety.reopen")
    guard = WarnOnce(limit=2, what="failures")

    with caplog.at_level(logging.INFO, logger=log.name):
        assert guard.first("a", log) and guard.first("b", log)
        assert guard.first("c", log) is False  # overflow notice
        for cycle in range(5):
            guard.forget("a")
            assert guard.first(f"new-{cycle}", log) is False

    assert len(_warnings(caplog)) == 1


def test_warn_once_starts_over_after_the_window(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = logging.getLogger("headroom.test_log_safety.window")
    guard = WarnOnce(limit=2, what="hook failures", window_seconds=600)
    clock = [1000.0]
    monkeypatch.setattr(log_safety.time, "monotonic", lambda: clock[0])

    results = []
    with caplog.at_level(logging.INFO, logger=log.name):
        steps = (("a", 0), ("b", 0), ("c", 0), ("a", 300), ("d", 299), ("a", 1), ("new-hook", 0))
        for key, advance in steps:
            clock[0] += advance
            results.append(guard.first(key, log))

    # a and b warn and c is the overflow notice; inside the window a stays quiet
    # and d is suppressed; once the window ends, the still-recurring a and a
    # newly broken key both warn again.
    assert results == [True, True, False, False, False, True, True]
    assert len(_warnings(caplog)) == 1


def test_warn_once_needs_a_positive_window() -> None:
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        WarnOnce(limit=1, what="failures", window_seconds=0)


@pytest.mark.parametrize("digits", [19, 10_001])
def test_out_of_range_errno_never_breaks_the_description(digits: int) -> None:
    """An errno too large to print (Python caps int-to-text at 4300 digits) is not printed."""
    exc = OSError()
    exc.errno = 10 ** (digits - 1)  # computed, not a literal, so collection never formats it

    assert describe_exception(exc) == "OSError [Errno out of range]"


def test_negative_dns_error_codes_are_kept() -> None:
    import socket

    assert describe_exception(socket.gaierror(-2, "Name or service not known")) == (
        "gaierror [Errno -2]"
    )


def test_overflow_notice_states_the_time_left_in_the_window(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = logging.getLogger("headroom.test_log_safety.remaining")
    guard = WarnOnce(limit=1, what="failures", window_seconds=3600)
    clock = [0.0]
    monkeypatch.setattr(log_safety.time, "monotonic", lambda: clock[0])

    with caplog.at_level(logging.INFO, logger=log.name):
        guard.first("a", log)
        clock[0] = 3000.0  # ten minutes left in the window
        guard.first("b", log)

    assert _warnings(caplog)[-1].endswith("for the next 10 minutes")


def test_forget_cycles_cannot_exceed_the_window_budget(caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("headroom.test_log_safety.forget_budget")
    guard = WarnOnce(limit=3, what="failures")

    with caplog.at_level(logging.INFO, logger=log.name):
        issued = 0
        for cycle in range(50):
            if guard.first(f"path-{cycle}", log):
                issued += 1
                log.warning("path-%d failed", cycle)
            guard.forget(f"path-{cycle}")

    assert issued == 3
    assert len(_warnings(caplog)) == 4  # three warnings plus one overflow notice


@pytest.mark.parametrize(
    ("filename", "module_name"),
    [
        ("sk-FILE-CANARY\nforged", None),
        (os.__file__, "os"),
        (log_safety.__file__, "headroom.log_safety"),
    ],
    ids=["forged-name", "claims-stdlib-file", "claims-headroom-module"],
)
def test_runtime_compiled_frames_are_not_logged(
    monkeypatch: pytest.MonkeyPatch, filename: str, module_name: str | None
) -> None:
    """Runtime code may claim a real file and module name; its own names still stay out."""
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    code = compile("def sk_FUNC_CANARY():\n    raise ValueError('x')\n", filename, "exec")
    namespace: dict[str, object] = {} if module_name is None else {"__name__": module_name}
    exec(code, namespace)
    with pytest.raises(ValueError) as caught:
        namespace["sk_FUNC_CANARY"]()  # type: ignore[operator]

    text = describe_exception(caught.value)
    assert "CANARY" not in text
    assert "<dynamic code>" in text
    assert "test_log_safety.py:" in text  # the calling test frame keeps its location


def test_zip_and_sourceless_modules_keep_their_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "def boom():\n    raise ValueError('x')\n"
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as bundle:
        bundle.writestr("zipped_mod.py", source)
    (tmp_path / "pyc_mod.py").write_text(source)
    py_compile.compile(str(tmp_path / "pyc_mod.py"), cfile=str(tmp_path / "pyc_mod.pyc"))
    (tmp_path / "pyc_mod.py").unlink()
    monkeypatch.syspath_prepend(str(tmp_path / "bundle.zip"))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for name in ("zipped_mod", "pyc_mod"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    texts = []
    for name in ("zipped_mod", "pyc_mod"):
        module = importlib.import_module(name)
        with pytest.raises(ValueError) as caught:
            module.boom()
        texts.append(describe_exception(caught.value))

    assert "zipped_mod.py:2 in boom" in texts[0]
    assert "pyc_mod.pyc:2 in boom" in texts[1]


@pytest.mark.parametrize(
    "value",
    ["sk-ant-abc123:x", "AKIAABCDEFGH:secret", "myuser:hunter2@db:5432", "AKIASECRET://host/x"],
)
def test_text_that_is_not_a_url_never_reaches_the_log(value: str) -> None:
    text = redact_url(value)
    for secret in ("sk-ant", "abc123", "AKIA", "akia", "hunter2", "myuser", "SECRET", "secret"):
        assert secret not in text


@pytest.mark.parametrize("host", ["a b", "a\x0bb\x0cc", "a\x85b"])
def test_hosts_with_line_breaks_are_not_logged(host: str) -> None:
    assert redact_url(f"http://{host}/") == "<unparseable url>"


def test_class_names_cannot_forge_a_log_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    forged = type("Err\nWARNING forged sk-QUAL", (Exception,), {})

    text = describe_exception(forged("x"))

    assert "\n" not in text and "sk-QUAL" not in text
    assert text.startswith("<exception>")


def test_odd_errno_and_module_objects_never_break_the_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)

    class FormatsItself(int):
        def __format__(self, spec: str) -> str:
            return "sk-ERRNO-CANARY"

    exc = OSError()
    exc.errno = FormatsItself(2)
    assert "CANARY" not in describe_exception(exc)

    class RaisingGet(dict):  # type: ignore[type-arg]
        def get(self, *args: object) -> object:
            raise RuntimeError("get")

    namespace = RaisingGet()
    exec(compile("def boom():\n    raise ValueError('x')\n", "<x>", "exec"), namespace)
    with pytest.raises(ValueError) as caught:
        namespace["boom"]()
    assert "<dynamic code>" in describe_exception(caught.value)

    lazy = types.ModuleType("lazy_sk_mod")

    def lazy_getattr(attr: str) -> str:
        return "sk-FILE-CANARY.py"

    lazy.__getattr__ = lazy_getattr  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "lazy_sk_mod", lazy)
    exec(compile("def boom():\n    raise ValueError('x')\n", "<x>", "exec"), vars(lazy))
    with pytest.raises(ValueError) as caught:
        lazy.boom()
    assert "CANARY" not in describe_exception(caught.value)


def test_safe_id_never_raises() -> None:
    class BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr")

    huge = 10 ** int("5000")
    assert safe_id(huge) == "<unrepresentable id>"
    assert safe_id(BadRepr()) == "<unrepresentable id>"


@pytest.mark.parametrize("window", [math.inf, math.nan, 0, -1])
def test_warn_once_needs_a_finite_positive_window(window: float) -> None:
    with pytest.raises(ValueError, match="window_seconds must be positive and finite"):
        WarnOnce(limit=1, what="failures", window_seconds=window)
