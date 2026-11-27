"""uvicorn's logging config, with every record kept on one line.

uvicorn configures its own ``uvicorn`` and ``uvicorn.access`` loggers with
``propagate: False``, so their records (access lines carrying the request
path, "Exception in ASGI application" tracebacks that can quote request
text) never reach the proxy's own handlers. This is uvicorn's default config
with the same format strings, handlers and levels, but formatters that escape
control characters and indent tracebacks like ``SingleLineFormatter``.
"""

from __future__ import annotations

import copy
from typing import Any

from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter, DefaultFormatter

from headroom.exceptions import ConfigurationError
from headroom.log_safety import SingleLineFormatting


class SingleLineDefaultFormatter(SingleLineFormatting, DefaultFormatter):
    """uvicorn's default (coloured level prefix) formatter, one line per record."""


class SingleLineAccessFormatter(SingleLineFormatting, AccessFormatter):
    """uvicorn's access-log formatter, one line per record."""


_FORMATTERS = {
    "uvicorn.logging.DefaultFormatter": f"{__name__}.SingleLineDefaultFormatter",
    "uvicorn.logging.AccessFormatter": f"{__name__}.SingleLineAccessFormatter",
}


def uvicorn_log_config() -> dict[str, Any]:
    """uvicorn's default ``log_config`` with the single-line formatters swapped in."""
    # Annotated: where uvicorn is not installed (the lint job), mypy sees it as Any.
    config: dict[str, Any] = copy.deepcopy(LOGGING_CONFIG)
    for formatter in config["formatters"].values():
        factory = formatter.get("()")
        if factory not in _FORMATTERS:
            raise ConfigurationError(
                f"uvicorn's default log formatter {factory!r} is not one Headroom "
                "knows how to keep single-line; update headroom/proxy/uvicorn_logging.py."
            )
        formatter["()"] = _FORMATTERS[factory]
    return config
