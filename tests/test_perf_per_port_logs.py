"""Per-port and per-worker proxy log files.

Every proxy writes ``proxy-<port>.log``; multi-worker deployments add the PID
so same-port workers do not share a ``RotatingFileHandler`` target and race
during rollover. The perf reader aggregates worker files, per-port files, and
the legacy shared ``proxy.log``.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from headroom import paths as _paths
from headroom.cli import wrap as wrap_cli
from headroom.perf import analyzer
from headroom.proxy import server
from headroom.proxy.helpers import _setup_file_logging


def _perf_line(ts: str, rid: str, model: str) -> str:
    return (
        f"{ts} - headroom.proxy - INFO - [{rid}] PERF "
        f"model={model} msgs=3 tok_before=1000 tok_after=400 "
        f"tok_saved=600 cache_read=0 cache_write=0 cache_hit_pct=0 "
        f"opt_ms=1 transforms=agent90_smoke client=test"
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_proxy_log_path_is_per_port() -> None:
    a = _paths.proxy_log_path(8888)
    b = _paths.proxy_log_path(8889)
    assert a.name == "proxy-8888.log"
    assert b.name == "proxy-8889.log"
    assert a != b  # the core anti-collision property
    # Legacy shared name preserved for the readers' fallback.
    assert _paths.proxy_log_path().name == "proxy.log"


def test_proxy_log_path_is_per_worker() -> None:
    assert _paths.proxy_log_path(8888, process_id=1234).name == "proxy-8888-1234.log"
    assert _paths.proxy_log_path(8888, process_id=5678).name == "proxy-8888-5678.log"


def test_stdio_log_path_is_per_port() -> None:
    assert _paths.proxy_stdio_log_path(8888).name == "proxy-stdio-8888.log"
    assert _paths.proxy_stdio_log_path().name == "proxy-stdio.log"
    # wrap helper agrees and keeps stdio beside the runtime log.
    stdio = wrap_cli._get_proxy_stdio_log_path(8888)
    assert stdio.name == "proxy-stdio-8888.log"
    assert stdio.parent == wrap_cli._get_log_path(8888).parent


def test_setup_file_logging_targets_per_port_file(workspace: Path) -> None:
    logger = logging.getLogger("headroom")
    original = list(logger.handlers)
    for h in original:
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
    try:
        _setup_file_logging(8888)
        _setup_file_logging(8888)
        rotating = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        assert Path(rotating[-1].baseFilename).name == "proxy-8888.log"
    finally:
        for h in [x for x in logger.handlers if isinstance(x, RotatingFileHandler)]:
            h.close()
            logger.removeHandler(h)
        for h in original:
            logger.addHandler(h)


def test_setup_file_logging_keeps_root_propagation_and_writes_once_per_sink(
    workspace: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("headroom")
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    for handler in original_handlers:
        logger.removeHandler(handler)

    try:
        logger.propagate = True
        with caplog.at_level(logging.INFO):
            _setup_file_logging(8888)
            logger.info("propagation-regression-record")

        for handler in logger.handlers:
            handler.flush()

        propagated = [
            record
            for record in caplog.records
            if record.getMessage() == "propagation-regression-record"
        ]
        file_text = (workspace / "logs" / "proxy-8888.log").read_text()
        assert len(propagated) == 1
        assert file_text.count("propagation-regression-record") == 1
    finally:
        for handler in list(logger.handlers):
            if isinstance(handler, RotatingFileHandler):
                handler.close()
                logger.removeHandler(handler)
        logger.propagate = original_propagate
        for handler in original_handlers:
            logger.addHandler(handler)


def test_setup_file_logging_targets_worker_file(workspace: Path) -> None:
    logger = logging.getLogger("headroom")
    original = list(logger.handlers)
    for handler in original:
        if isinstance(handler, RotatingFileHandler):
            logger.removeHandler(handler)
    try:
        _setup_file_logging(8888, process_id=1234)
        rotating = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        assert Path(rotating[0].baseFilename).name == "proxy-8888-1234.log"
    finally:
        for handler in [x for x in logger.handlers if isinstance(x, RotatingFileHandler)]:
            handler.close()
            logger.removeHandler(handler)
        for handler in original:
            logger.addHandler(handler)


def test_setup_file_logging_reconfigures_for_sequential_port_change(
    workspace: Path,
) -> None:
    """Sequential app creation closes the old handler and selects the new port.

    Regression: when the ``RotatingFileHandler`` was constructed before the
    dedup guard, the second ``_setup_file_logging(port)`` left an empty stray
    ``proxy-<port>.log``, leaked the handler fd, and routed the new port's
    records into the first port's file.
    """
    log_dir = workspace / "logs"
    logger = logging.getLogger("headroom")
    original = list(logger.handlers)
    for h in original:
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
    try:
        _setup_file_logging(8888)
        logger.info("record-for-8888")
        _setup_file_logging(9999)  # a second proxy/app in the same process
        logger.info("record-for-9999")

        # Exactly one file handler is ever attached, now pointing at 9999.
        rotating = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        assert Path(rotating[0].baseFilename).name == "proxy-9999.log"
        rotating[0].flush()

        text_8888 = (log_dir / "proxy-8888.log").read_text()
        text_9999 = (log_dir / "proxy-9999.log").read_text()
        # The records written before and after reconfiguration use their
        # respective paths, and the first file is not an empty stray.
        assert "record-for-8888" in text_8888 and "record-for-9999" not in text_8888
        assert "record-for-9999" in text_9999 and "record-for-8888" not in text_9999
    finally:
        for h in [x for x in logger.handlers if isinstance(x, RotatingFileHandler)]:
            h.close()
            logger.removeHandler(h)
        for h in original:
            logger.addHandler(h)


def test_setup_file_logging_preserves_external_rotating_handler(workspace: Path) -> None:
    logger = logging.getLogger("headroom")
    original = list(logger.handlers)
    for handler in original:
        logger.removeHandler(handler)
    external = RotatingFileHandler(workspace / "logs" / "external.log")
    logger.addHandler(external)
    try:
        _setup_file_logging(8888)
        _setup_file_logging(9999)

        assert external in logger.handlers
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        for handler in original:
            logger.addHandler(handler)


def test_create_app_default_config_keys_logging_by_default_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int | None]] = []
    monkeypatch.setattr(
        server,
        "_setup_file_logging",
        lambda port, process_id=None: calls.append((port, process_id)),
    )

    app = server.create_app()

    assert app.state.proxy.config.port == 8787
    assert calls == [(8787, None)]


def test_create_app_keys_multi_worker_logging_by_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int | None]] = []
    monkeypatch.setattr(
        server,
        "_setup_file_logging",
        lambda port, process_id=None: calls.append((port, process_id)),
    )
    monkeypatch.setattr(server.os, "getpid", lambda: 1234)

    server.create_app(server.ProxyConfig(port=8888, worker_processes=2))

    assert calls == [(8888, 1234)]


def test_parse_log_files_aggregates_per_port_and_legacy(workspace: Path) -> None:
    log_dir = workspace / "logs"
    ts = "2026-08-22 10:00:00,000"
    (log_dir / "proxy-8888-1111.log").write_text(_perf_line(ts, "hr_a", "model-A") + "\n")
    (log_dir / "proxy-8888-2222.log").write_text(_perf_line(ts, "hr_b", "model-B") + "\n")
    (log_dir / "proxy-8888-1111.log.1").write_text(_perf_line(ts, "hr_e", "model-E") + "\n")
    (log_dir / "proxy-8889.log").write_text(_perf_line(ts, "hr_c", "model-C") + "\n")
    (log_dir / "proxy.log").write_text(_perf_line(ts, "hr_d", "model-D") + "\n")
    # stdio captures and any other non-PERF proxy-*.log are excluded by the
    # positive filename filter, not just a proxy-stdio blacklist.
    (log_dir / "proxy-stdio-8888.log").write_text(_perf_line(ts, "hr_stdio", "model-STDIO") + "\n")
    (log_dir / "proxy-stdio.log").write_text(_perf_line(ts, "hr_stdio2", "model-STDIO2") + "\n")
    (log_dir / "proxy-errors.log").write_text(_perf_line(ts, "hr_err", "model-ERR") + "\n")

    report = analyzer.parse_log_files(last_n_hours=0.0)  # 0 => no cutoff, all data
    models = {r.model for r in report.perf_records}
    rids = {r.request_id for r in report.perf_records}

    assert {"model-A", "model-B", "model-C", "model-D", "model-E"} <= models
    # Non-PERF files never ingested.
    assert models.isdisjoint({"model-STDIO", "model-STDIO2", "model-ERR"})
    assert rids.isdisjoint({"hr_stdio", "hr_stdio2", "hr_err"})


@pytest.fixture
def headroom_log_state(request: pytest.FixtureRequest):
    """Restore the headroom logger level and drop the debug filter from root handlers."""
    from headroom.proxy import helpers

    headroom_logger = logging.getLogger("headroom")
    level = headroom_logger.level

    def restore() -> None:
        headroom_logger.setLevel(level)
        loggers = [logging.getLogger(), headroom_logger] + [
            lg
            for name, lg in logging.Logger.manager.loggerDict.items()
            if name.startswith("headroom.") and isinstance(lg, logging.Logger)
        ]
        for handler in [h for lg in loggers for h in lg.handlers]:
            for f in list(handler.filters):
                if isinstance(f, helpers._HeadroomDebugStaysInProxyLog):
                    handler.removeFilter(f)

    request.addfinalizer(restore)
    return headroom_logger


def _flush(headroom_logger: logging.Logger) -> None:
    for handler in headroom_logger.handlers:
        handler.flush()


@pytest.mark.parametrize(
    ("env", "debug_written"),
    [("debug", True), ("TRACE", True), ("warning", False), ("info", False), (None, False)],
)
def test_headroom_log_level_debug_reaches_proxy_log(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    headroom_log_state: logging.Logger,
    env: str | None,
    debug_written: bool,
) -> None:
    """``HEADROOM_LOG_LEVEL=debug`` must surface Headroom's own debug lines, not only uvicorn's."""
    if env is None:
        monkeypatch.delenv("HEADROOM_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("HEADROOM_LOG_LEVEL", env)

    _setup_file_logging(port=9911)
    probe = logging.getLogger("headroom.proxy.log_level_probe")
    probe.debug("debug-probe")
    probe.info("info-probe")
    _flush(headroom_log_state)

    text = _paths.proxy_log_path(9911).read_text()
    assert "info-probe" in text
    assert ("debug-probe" in text) is debug_written
    assert ("Headroom debug logging is on" in text) is debug_written


def test_headroom_debug_stays_out_of_stdout(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, headroom_log_state: logging.Logger
) -> None:
    """Debug lines can carry tool output: only the owner-only proxy.log may receive them."""
    import io

    stdout = io.StringIO()
    root_handler = logging.StreamHandler(stdout)
    logging.getLogger().addHandler(root_handler)
    try:
        monkeypatch.setenv("HEADROOM_LOG_LEVEL", "debug")
        _setup_file_logging(port=9912)
        probe = logging.getLogger("headroom.transforms.log_level_probe")
        probe.debug("secret-tool-output")
        probe.info("info-probe")
        logging.getLogger("thirdparty.log_level_probe").warning("thirdparty-warning")
        _flush(headroom_log_state)
    finally:
        logging.getLogger().removeHandler(root_handler)

    assert "secret-tool-output" in _paths.proxy_log_path(9912).read_text()
    assert "secret-tool-output" not in stdout.getvalue()
    assert "info-probe" in stdout.getvalue()
    assert "thirdparty-warning" in stdout.getvalue()


def test_reused_proxy_log_picks_up_debug(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, headroom_log_state: logging.Logger
) -> None:
    """A second same-port setup after switching to debug must not keep the old INFO handler."""
    monkeypatch.setenv("HEADROOM_LOG_LEVEL", "info")
    _setup_file_logging(port=9913)
    monkeypatch.setenv("HEADROOM_LOG_LEVEL", "debug")
    _setup_file_logging(port=9913)

    logging.getLogger("headroom.proxy.log_level_probe").debug("debug-after-reuse")
    _flush(headroom_log_state)

    assert "debug-after-reuse" in _paths.proxy_log_path(9913).read_text()


def test_headroom_debug_stays_out_of_child_logger_handlers(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, headroom_log_state: logging.Logger
) -> None:
    """A handler on a headroom.* logger (e.g. an extension's) must not receive debug lines either."""
    import io

    stream = io.StringIO()
    child = logging.getLogger("headroom.extension_probe")
    child_handler = logging.StreamHandler(stream)
    child.addHandler(child_handler)
    try:
        monkeypatch.setenv("HEADROOM_LOG_LEVEL", "debug")
        _setup_file_logging(port=9914)
        child.debug("secret-tool-output")
        child.info("info-probe")
        _flush(headroom_log_state)
    finally:
        child.removeHandler(child_handler)

    assert "secret-tool-output" in _paths.proxy_log_path(9914).read_text()
    assert "secret-tool-output" not in stream.getvalue()
    assert "info-probe" in stream.getvalue()


@pytest.mark.parametrize(("dump", "logged"), [("", False), ("full", True)])
def test_router_content_dump_needs_the_content_opt_in(
    monkeypatch: pytest.MonkeyPatch, headroom_log_state: logging.Logger, dump: str, logged: bool
) -> None:
    """HEADROOM_LOG_LEVEL=debug alone must not write the tool output the router compresses."""
    import io

    from headroom.transforms import content_router

    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", dump)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    router_logger = logging.getLogger(content_router.__name__)
    previous = router_logger.level
    router_logger.addHandler(handler)
    router_logger.setLevel(logging.DEBUG)
    try:
        content_router._log_router_debug("routing_decision", content="secret-tool-output")
    finally:
        router_logger.removeHandler(handler)
        router_logger.setLevel(previous)

    assert ("secret-tool-output" in stream.getvalue()) is logged
