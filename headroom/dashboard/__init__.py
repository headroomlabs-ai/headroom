"""Headroom Dashboard - Real-time proxy monitoring UI."""

from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
# Vendored tailwind/htmx/alpine. Served locally because Edge's Tracking
# Prevention and corporate proxies block unpkg.com/cdn.tailwindcss.com, which
# left the dashboard unstyled and dataless on some Windows machines.
STATIC_DIR = DASHBOARD_DIR / "static"


def get_dashboard_html() -> str:
    """Load the dashboard HTML template."""
    template_path = TEMPLATES_DIR / "dashboard.html"
    return template_path.read_text(encoding="utf-8")


def get_settings_html() -> str:
    """Load the settings GUI HTML template."""
    template_path = TEMPLATES_DIR / "settings.html"
    return template_path.read_text(encoding="utf-8")
