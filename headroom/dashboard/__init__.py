"""Headroom Dashboard - Real-time proxy monitoring UI."""

import mimetypes
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
# Vendored tailwind/htmx/alpine. Served locally because Edge's Tracking
# Prevention and corporate proxies block unpkg.com/cdn.tailwindcss.com, which
# left the dashboard unstyled and dataless on some Windows machines.
STATIC_DIR = DASHBOARD_DIR / "static"


#: Correct types for the asset kinds the dashboard mount can serve. Values are
#: the current IANA/WHATWG registrations, so this only ever repairs a host
#: database — it never invents a mapping of our own.
_STATIC_MIME_TYPES: tuple[tuple[str, str], ...] = (
    ("text/javascript", ".js"),
    ("text/javascript", ".mjs"),
    ("text/css", ".css"),
    ("application/json", ".json"),
    # Source maps are JSON documents even though they accompany .js assets.
    ("application/json", ".map"),
)


def register_static_mime_types() -> None:
    r"""Pin the MIME types used for the vendored dashboard assets.

    ``StaticFiles`` derives ``Content-Type`` from :func:`mimetypes.guess_type`,
    and Python seeds that database from the host: the Windows registry
    (``HKCR\<ext>\Content Type``) and, elsewhere, files like
    ``/etc/mime.types``. A host that maps ``.js`` to ``text/plain`` — a stale
    registry entry, or a minimal container image carrying no mime database at
    all — makes the proxy serve ``alpine.min.js`` and its siblings as plain
    text. The proxy also sends ``X-Content-Type-Options: nosniff`` on every
    response, so the browser refuses to execute a script labelled that way and
    the dashboard loads unstyled and dataless (#3179).

    Registering the standard mappings makes the served type independent of the
    host database. :func:`mimetypes.add_type` is strict by default, so these
    replace a bad host entry rather than losing to it.

    Called from ``create_app`` rather than at import time: correcting the
    process-wide table is right for the proxy that serves these files, but it
    is not a side effect ``import headroom`` should have on a library consumer.
    """
    for mime_type, extension in _STATIC_MIME_TYPES:
        mimetypes.add_type(mime_type, extension)


def get_dashboard_html() -> str:
    """Load the dashboard HTML template."""
    template_path = TEMPLATES_DIR / "dashboard.html"
    return template_path.read_text(encoding="utf-8")


def get_settings_html() -> str:
    """Load the settings GUI HTML template."""
    template_path = TEMPLATES_DIR / "settings.html"
    return template_path.read_text(encoding="utf-8")
