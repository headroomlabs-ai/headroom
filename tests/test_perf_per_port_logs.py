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
