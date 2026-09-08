"""Dashboard assets must be served as JavaScript even where the host disagrees.

``StaticFiles`` types every response from :func:`mimetypes.guess_type`, and
Python seeds that database from the host — the Windows registry, or files like
``/etc/mime.types``. Paired with the proxy's unconditional
``X-Content-Type-Options: nosniff``, a host that calls ``.js`` ``text/plain``
stops the browser executing the dashboard entirely (#3179).
"""

from __future__ import annotations

import mimetypes

import pytest

pytest.importorskip("starlette")

from starlette.applications import Starlette  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from headroom.dashboard import (  # noqa: E402
    _STATIC_MIME_TYPES,
    STATIC_DIR,
    register_static_mime_types,
)

ASSETS = ["tailwind.min.js", "htmx.min.js", "alpine.min.js"]


@pytest.fixture(autouse=True)
def _restore_mime_db():
    """Rebuild the process-wide table afterwards — ``add_type`` mutates it."""
    yield
    mimetypes.init()


def _break_host_db(*extensions: str) -> None:
    """Simulate a host whose mime database calls these extensions plain text."""
    for extension in extensions or (".js",):
        mimetypes.add_type("text/plain", extension)


def _static_client() -> TestClient:
    """Mount the real vendored assets exactly as ``create_app`` does."""
    app = Starlette()
    app.mount("/dashboard/static", StaticFiles(directory=STATIC_DIR, check_dir=False))
    return TestClient(app)


@pytest.mark.parametrize("asset", ASSETS)
def test_asset_is_javascript_on_a_broken_host(asset: str) -> None:
    _break_host_db()
    register_static_mime_types()

    with _static_client() as client:
        resp = client.get(f"/dashboard/static/{asset}")

    assert resp.status_code == 200, resp.text
    # Under nosniff the browser executes nothing that is not a JavaScript type.
    assert "javascript" in resp.headers["content-type"]


def test_registration_overrides_a_bad_host_mapping() -> None:
    """``add_type`` is strict by default, so it must win against the host."""
    _break_host_db()
    assert mimetypes.guess_type("alpine.min.js")[0] == "text/plain"

    register_static_mime_types()

    assert mimetypes.guess_type("alpine.min.js")[0] == "text/javascript"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("app.js", "text/javascript"),
        ("module.mjs", "text/javascript"),
        ("styles.css", "text/css"),
        ("data.json", "application/json"),
        # A source map is a JSON document, despite shipping beside .js assets.
        ("bundle.js.map", "application/json"),
    ],
)
def test_each_registered_extension_resolves(filename: str, expected: str) -> None:
    _break_host_db(*(extension for _type, extension in _STATIC_MIME_TYPES))

    register_static_mime_types()

    assert mimetypes.guess_type(filename)[0] == expected


def test_registration_is_idempotent() -> None:
    """``create_app`` may run more than once in a process (tests, embedding)."""
    _break_host_db()
    register_static_mime_types()
    register_static_mime_types()

    assert mimetypes.guess_type("alpine.min.js")[0] == "text/javascript"


def test_every_vendored_asset_extension_is_registered() -> None:
    """A newly vendored asset kind must not silently inherit the host database."""
    registered = {extension for _type, extension in _STATIC_MIME_TYPES}
    unregistered = sorted(
        {
            path.suffix
            for path in STATIC_DIR.iterdir()
            if path.is_file() and path.suffix not in registered
        }
    )

    assert unregistered == []
