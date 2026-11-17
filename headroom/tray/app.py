"""System tray app (pystray). Needs the optional ``headroom-ai[tray]`` extra."""

from __future__ import annotations

import sys
import threading
import webbrowser
from collections.abc import Callable
from typing import Any

from . import core
from .icon import render_icon

# Exit code the tray uses to ask its owning ``headroom proxy`` to shut down.
QUIT_EXIT_CODE = 64
RELEASES_URL = "https://github.com/headroomlabs-ai/headroom/releases/latest"
POLL_SECONDS = 5.0


def tray_available() -> tuple[bool, str]:
    """Whether the tray can run here, and why not when it can't."""
    if sys.platform != "win32":
        return False, "the system tray is currently supported on Windows only"
    try:
        import PIL  # noqa: F401
        import pystray  # noqa: F401
    except ImportError:
        return False, "missing dependency; install with: pip install 'headroom-ai[tray]'"
    return True, ""


class _State:
    def __init__(self) -> None:
        self.snap = core.Snapshot()
        self.update: str | None = None
        self.notified: str | None = None
        self.exit_code = 0


def run_tray(
    url: str,
    *,
    owns_proxy: bool = False,
    poll_seconds: float = POLL_SECONDS,
    on_ready: Callable[[], None] | None = None,
) -> int:
    """Run the tray until the user exits. Returns the process exit code."""
    import pystray

    state = _State()
    stop = threading.Event()

    def open_dashboard(icon: Any = None, item: Any = None) -> None:
        webbrowser.open(core.dashboard_url(url))

    def open_release(icon: Any = None, item: Any = None) -> None:
        webbrowser.open(RELEASES_URL)

    def quit_app(icon: Any, item: Any = None) -> None:
        state.exit_code = QUIT_EXIT_CODE if owns_proxy else 0
        stop.set()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Headroom", None, enabled=False),
        pystray.MenuItem(lambda _i: core.stats_label(state.snap), open_dashboard),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Dashboard", open_dashboard, default=True),
        pystray.MenuItem(
            lambda _i: f"Update available: {state.update}",
            open_release,
            visible=lambda _i: bool(state.update),
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit Headroom" if owns_proxy else "Exit", quit_app),
    )
    icon = pystray.Icon("headroom", render_icon(online=False), "Headroom", menu)

    def refresh() -> None:
        snap, update = core.fetch_snapshot(url), core.available_update()
        state.snap, state.update = snap, update
        icon.icon = render_icon(online=snap.online, update=bool(update))
        icon.title = core.tooltip(snap, update)
        icon.update_menu()
        if update and state.notified != update:
            state.notified = update
            try:
                icon.notify(
                    f"Headroom {update} is available. Run: headroom update", "Headroom update"
                )
            except Exception:
                pass

    def poll_loop() -> None:
        while not stop.is_set():
            try:
                refresh()
            except Exception:
                pass
            stop.wait(poll_seconds)

    def setup(icon: Any) -> None:
        icon.visible = True
        threading.Thread(target=poll_loop, name="headroom-tray-poll", daemon=True).start()
        if on_ready:
            on_ready()

    icon.run(setup)
    stop.set()
    return state.exit_code
