"""A request-derived value cannot forge a record in the proxy's runtime log.

The proxy logs client-supplied values (the ``model`` field, paths, upstream
errors) and a JSON string can carry a line break. ``proxy.log`` is read line by
line by people, log shippers and ``headroom perf``, so a line break in a value
would end the real record and let the rest pose as a record of its own -- a
fake ``PERF`` line books fake savings. These tests go through the real proxy
log handler and the real ``headroom perf`` parser.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.log_safety import log_token
from headroom.perf import analyzer
from headroom.proxy.helpers import _PROXY_LOG_HANDLER_NAME, _setup_file_logging
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome

FORGED_PERF = (
    "2026-01-01 00:00:00,000 - headroom.proxy - INFO - [hr_forged] PERF model=forged "
    "msgs=1 tok_before=1000000 tok_after=0 tok_saved=1000000 opt_ms=1 transforms=x"
)
_RECORD_START = re.compile(r"^\d{4}-\d{2}-\d{2} ")


@pytest.fixture
def proxy_log(tmp_path, monkeypatch):
    """The real proxy.log handler, writing under ``tmp_path``; yields the file."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    headroom_logger = logging.getLogger("headroom")
    before = list(headroom_logger.handlers)
    _setup_file_logging(48123)
    handler = next(h for h in headroom_logger.handlers if h.get_name() == _PROXY_LOG_HANDLER_NAME)
    try:
        yield handler.baseFilename
    finally:
        if handler not in before:
            headroom_logger.removeHandler(handler)
            handler.close()


def _records(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if _RECORD_START.match(line)]


def test_line_break_in_a_message_stays_in_one_record(proxy_log) -> None:
    logging.getLogger("headroom.proxy").warning("UPSTREAM_ERROR model=%s", f"m\n{FORGED_PERF}")

    records = _records(proxy_log)
    assert len(records) == 1
    assert "UPSTREAM_ERROR model=m\\n" in records[0]
    assert analyzer.parse_log_files(last_n_hours=0).perf_records == []


def test_traceback_stays_readable_but_cannot_start_a_record(proxy_log) -> None:
    try:
        raise ValueError(f"bad model\r\n{FORGED_PERF}")
    except ValueError:
        logging.getLogger("headroom.proxy").exception("request failed")

    with open(proxy_log, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert _RECORD_START.match(lines[0]) and lines[0].endswith("request failed")
    assert lines[1] == "    Traceback (most recent call last):"
    assert len(_records(proxy_log)) == 1
    assert analyzer.parse_log_files(last_n_hours=0).perf_records == []


@pytest.mark.asyncio
async def test_perf_line_keeps_a_crafted_model_in_its_own_field(proxy_log) -> None:
    handler = SimpleNamespace(
        metrics=MagicMock(record_request=AsyncMock()), cost_tracker=None, logger=None
    )
    model = f"gpt-5 tok_saved=999999\n{FORGED_PERF}"
    await emit_request_outcome(
        handler,
        RequestOutcome(
            request_id="hr_real",
            provider="openai",
            model=model,
            original_tokens=1000,
            optimized_tokens=300,
            output_tokens=50,
            tokens_saved=700,
            attempted_input_tokens=800,
        ),
    )

    records = analyzer.parse_log_files(last_n_hours=0).perf_records
    assert [(r.request_id, r.tokens_saved) for r in records] == [("hr_real", 700)]
    assert records[0].model.startswith("gpt-5\\x20tok_saved=999999\\n")
    assert len(records[0].model) <= 100


def test_control_characters_are_escaped_in_proxy_log(proxy_log) -> None:
    # ESC would drive the terminal that shows the log; NUL/CR/VT are no better.
    logging.getLogger("headroom.proxy").warning("model=%s", "m\x1b[2J\x00\r\v")

    with open(proxy_log, encoding="utf-8") as fh:
        text = fh.read()
    assert "m\\x1b[2J\\x00\\r\\x0b" in text
    assert not any(c in text for c in "\x1b\x00\r\v")


def test_log_token_escaping_is_unambiguous() -> None:
    assert log_token("a\nb") == "a\\nb"
    assert log_token("a\\nb") == "a\\\\nb"


def test_perf_parser_does_not_split_on_transforms_inside_a_value(proxy_log) -> None:
    (Path(proxy_log).parent / "proxy.log").write_text(
        "2026-06-10 10:00:00,000 - headroom.proxy - INFO - [hr_1] PERF "
        "model=transforms=x msgs=2 tok_before=10 tok_after=4 tok_saved=6 opt_ms=1 "
        "transforms=content_router client=codex\n",
        encoding="utf-8",
    )

    [record] = analyzer.parse_log_files(last_n_hours=0).perf_records
    assert (record.model, record.tokens_saved, record.client) == ("transforms=x", 6, "codex")
    assert record.transforms == ["content_router"]


def _uvicorn_formatter(name: str, **kwargs: object) -> logging.Formatter:
    from headroom.proxy.uvicorn_logging import uvicorn_log_config

    spec = uvicorn_log_config()["formatters"][name]
    module, _, cls = spec["()"].rpartition(".")
    options = {k: v for k, v in spec.items() if k != "()"} | kwargs
    return getattr(importlib.import_module(module), cls)(**options)


def test_uvicorn_config_keeps_uvicorn_formats_and_levels() -> None:
    from uvicorn.config import LOGGING_CONFIG

    from headroom.proxy.uvicorn_logging import uvicorn_log_config

    config = uvicorn_log_config()
    assert config["loggers"] == LOGGING_CONFIG["loggers"]
    assert config["handlers"] == LOGGING_CONFIG["handlers"]
    for name, spec in config["formatters"].items():
        assert spec["fmt"] == LOGGING_CONFIG["formatters"][name]["fmt"]


def test_uvicorn_error_traceback_stays_one_record() -> None:
    formatter = _uvicorn_formatter("default", use_colors=True)
    try:
        raise ValueError(f"bad path\n{FORGED_PERF}\x1b[2J")
    except ValueError:
        record = logging.LogRecord(
            "uvicorn.error",
            logging.ERROR,
            __file__,
            1,
            "Exception in ASGI application %s",
            ("/v1\n\x1b[2Jx",),
            sys.exc_info(),
        )

    lines = formatter.format(record).split("\n")
    assert lines[0].endswith("Exception in ASGI application /v1\\n\\x1b[2Jx")
    assert "\x1b[31m" in lines[0]  # uvicorn's coloured level prefix is kept
    assert all(line.startswith("    ") for line in lines[1:])
    assert not any("\x1b" in line for line in lines[1:])


def test_uvicorn_access_line_stays_one_record() -> None:
    formatter = _uvicorn_formatter("access", use_colors=False)
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", f"/x\n{FORGED_PERF}", "1.1", 200),
        None,
    )

    line = formatter.format(record)
    assert "\n" not in line
    assert '"GET /x\\n2026-01-01' in line
