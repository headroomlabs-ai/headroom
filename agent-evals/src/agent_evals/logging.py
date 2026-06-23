"""Structured JSON logging. One configuration entry point; every event is a JSON line.

Attach structured fields via ``logger.info("msg", extra={"fields": {...}})``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_ROOT = "agent_evals"
_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the ``agent_evals`` logger tree. Idempotent."""

    global _configured
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger(_ROOT)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the tree with defaults on first use."""

    if not _configured:
        configure_logging()
    return logging.getLogger(f"{_ROOT}.{name}")
